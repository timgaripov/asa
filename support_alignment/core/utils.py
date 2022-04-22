from collections import OrderedDict
import hashlib
import io
from PIL import Image

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision.utils as vutils


class Logger(object):
    def __init__(self, log_dir, log_name='log.txt', verbose=False):
        self.log_dir = log_dir
        self.verbose = verbose
        self.log_file = open('%s/%s' % (log_dir, log_name), 'wt')

    def print(self, *objects):
        print(*objects, file=self.log_file, flush=True)
        if self.verbose:
            print(*objects)

    def __del__(self):
        self.log_file.close()


# Adapted from https://github.com/facebookresearch/DomainBed/blob/master/domainbed/lib/misc.py
def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def flatten_config_dict(config_dict):
    for key, value in config_dict.items():
        if isinstance(value, dict):
            for inner_name, inner_value in flatten_config_dict(value):
                yield f'{key}/{inner_name}', inner_value
        else:
            yield key, value


def evaluate(model, loader, weights, device):
    model.eval()

    classes = []
    predictions = []
    ces = []

    samples = None

    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        logits = model.predict(x)

        classes.extend(y.tolist())
        predictions.extend(torch.argmax(logits, dim=1).tolist())
        ces.extend(torch.nn.functional.cross_entropy(logits, y, reduction='none').tolist())

        if samples is None:
            samples = x.detach().cpu()

    model.train()

    classes = torch.tensor(classes)
    classes_one_hot = torch.nn.functional.one_hot(classes)
    class_sizes = torch.sum(classes_one_hot, dim=0)
    predictions = torch.tensor(predictions)
    indicators = torch.eq(classes, predictions).float()
    ces = torch.tensor(ces)
    if weights is None:
        weights = torch.ones_like(indicators)

    results = OrderedDict()
    results['accuracy'] = torch.mean(indicators).item() * 100.0
    results['accuracy_weight'] = (torch.sum(indicators * weights) / torch.sum(weights)).item() * 100.0
    class_accuracies = torch.sum(indicators[:, None] * classes_one_hot, dim=0) / class_sizes * 100.0
    results['accuracy_class_avg'] = torch.mean(class_accuracies).item()
    results['accuracy_class_min'] = torch.min(class_accuracies).item()

    results['ce'] = torch.mean(ces).item()
    results['ce_weight'] = (torch.sum(ces * weights) / torch.sum(weights)).item()
    class_ces = torch.sum(ces[:, None] * classes_one_hot, dim=0) / class_sizes
    results['ce_class_avg'] = torch.mean(class_ces).item()
    results['ce_class_min'] = torch.min(class_ces).item()

    return results, predictions.tolist(), classes.tolist(), samples


def batch_to_grid(img_batch, data_params=None):
    results = []
    img_batch = img_batch.detach().cpu()
    if data_params is not None:
        inv_norm_transform = data_params.get('inv_norm_transform', None)
        if inv_norm_transform is not None:
            img_batch = inv_norm_transform(img_batch)
    img_batch = torch.clamp(img_batch, 0.0, 1.0)
    samples = img_batch
    grid = vutils.make_grid(samples, nrow=8, normalize=True, range=(0.0, 1.0))
    np_image = np.transpose(grid.numpy(), [1, 2, 0])
    results.append(('batch', np_image))
    return results


# Adapted from https://github.com/timgaripov/swa/blob/4a2ddfdb2692eda91f2ac41533b62027976c605b/utils.py#L107
def update_bn(loader, model, device=None, sample_limit=50000):
    """Updates BN running statistics by feeding samples from the provided loader to the model."""
    was_training = model.training
    model.train()

    momenta = dict()
    track_stats = dict()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if module.training:
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum
                track_stats[module] = module.track_running_stats
                module.track_running_stats = True

    if not momenta:
        model.train(was_training)
        return

    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    with torch.no_grad():
        num_samples = 0
        for x in loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            if device is not None:
                x = x.to(device)

            model(x)

            num_samples += x.size(0)
            if num_samples > sample_limit:
                break

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    for bn_module in track_stats.keys():
        bn_module.track_running_stats = track_stats[bn_module]
    model.train(was_training)


def evaluate_alignment(feature_extractor, discriminator_list, names, train_iterator, num_batches):
    feature_extractor.train()
    for discriminator in discriminator_list:
        discriminator.train()

    outputs = [[torch.empty([0]), torch.empty([0])] for _ in range(len(discriminator_list))]
    for _ in range(num_batches):
        x_src, _, x_trg = next(train_iterator)

        with torch.no_grad():
            z_src = feature_extractor(x_src)
            z_trg = feature_extractor(x_trg)

            for i, discriminator in enumerate(discriminator_list):
                d_out_src = discriminator(z_src).view(-1)
                d_out_trg = discriminator(z_trg).view(-1)

                outputs[i][0] = torch.cat((outputs[i][0], d_out_src.detach().cpu()), dim=0)
                outputs[i][1] = torch.cat((outputs[i][1], d_out_trg.detach().cpu()), dim=0)

    metrics = OrderedDict()

    for i in range(len(outputs)):
        name = names[i]
        v_src, v_trg = outputs[i]

        metrics['{0}/num_samples'.format(name)] = v_src.size(0)

        vrange = max(1e-4, max(v_src.max().item(), v_trg.max().item()) - min(v_src.min().item(), v_trg.min().item()))
        metrics['{0}/range'.format(name)] = vrange

        metrics['{0}/src_min'.format(name)] = v_src.min().item()
        metrics['{0}/src_max'.format(name)] = v_src.max().item()
        metrics['{0}/trg_min'.format(name)] = v_trg.min().item()
        metrics['{0}/trg_max'.format(name)] = v_trg.max().item()

        v_all = torch.cat((v_src, v_trg), dim=0)
        y_all = torch.cat((torch.zeros_like(v_src), torch.ones_like(v_trg)), dim=0)
        log_loss = torch.nn.functional.binary_cross_entropy_with_logits(v_all, y_all)
        metrics['{0}/log_loss'.format(name)] = log_loss.item()

        v_src = v_src.view(-1, 1)
        v_trg = v_trg.view(-1, 1)

        v_src = v_src.view(-1)
        v_trg = v_trg.view(-1)

        v_src_sort, _ = torch.sort(v_src)
        v_trg_sort, _ = torch.sort(v_trg)
        ot_loss_abs = torch.mean(torch.abs(v_src_sort - v_trg_sort))
        ot_loss_sq = torch.mean(torch.square(v_src_sort - v_trg_sort))
        metrics['{0}/ot_abs'.format(name)] = ot_loss_abs.item()
        metrics['{0}/ot_rel'.format(name)] = ot_loss_abs.item() / vrange
        metrics['{0}/ot_sq'.format(name)] = ot_loss_sq.item()
        metrics['{0}/ot_rsq'.format(name)] = torch.sqrt(ot_loss_sq).item()
        metrics['{0}/ot_rsq_rel'.format(name)] = torch.sqrt(ot_loss_sq).item() / vrange

        if max(v_src.size(0), v_trg.size(0)) <= 10000:
            dist_abs_matrix = torch.abs(v_src[:, None] - v_trg[None, :])

            src_dist_abs_min, _ = torch.min(dist_abs_matrix, dim=1)
            trg_dist_abs_min, _ = torch.min(dist_abs_matrix, dim=0)
        else:
            k = 100
            src_dist_abs_min = torch.zeros(0)
            i = 0
            while i < v_src.size(0):
                dist_abs_matrix = torch.abs(v_src[i:i+k, None] - v_trg[None, :])
                src_dist_abs_min_cur, _ = torch.min(dist_abs_matrix, dim=1)
                src_dist_abs_min = torch.cat((src_dist_abs_min, src_dist_abs_min_cur))
                i += k

            trg_dist_abs_min = torch.zeros(0)
            i = 0
            while i < v_trg.size(0):
                dist_abs_matrix = torch.abs(v_src[:, None] - v_trg[None, i:i+k])
                trg_dist_abs_min_cur, _ = torch.min(dist_abs_matrix, dim=0)
                trg_dist_abs_min = torch.cat((trg_dist_abs_min, trg_dist_abs_min_cur))
                i += k

        src_dist_sq_min = torch.square(src_dist_abs_min)
        trg_dist_sq_min = torch.square(trg_dist_abs_min)

        supp_loss_abs_src = torch.mean(src_dist_abs_min)
        supp_loss_abs_trg = torch.mean(trg_dist_abs_min)
        supp_loss_sq_src = torch.mean(src_dist_sq_min)
        supp_loss_sq_trg = torch.mean(trg_dist_sq_min)
        supp_loss_abs = supp_loss_abs_src + supp_loss_abs_trg
        supp_loss_sq = supp_loss_sq_src + supp_loss_sq_trg
        metrics['{0}/supp_dist_abs'.format(name)] = supp_loss_abs.item()
        metrics['{0}/supp_dist_rel'.format(name)] = supp_loss_abs.item() / vrange
        metrics['{0}/supp_dist_sq'.format(name)] = supp_loss_sq.item()
        metrics['{0}/supp_dist_rsq'.format(name)] = torch.sqrt(supp_loss_sq).item()
        metrics['{0}/supp_dist_rsq_rel'.format(name)] = torch.sqrt(supp_loss_sq).item() / vrange

        metrics['{0}/supp_src_dist_abs'.format(name)] = supp_loss_abs_src.item()
        metrics['{0}/supp_src_dist_rel'.format(name)] = supp_loss_abs_src.item() / vrange
        metrics['{0}/supp_src_dist_sq'.format(name)] = supp_loss_sq_src.item()
        metrics['{0}/supp_src_dist_rsq'.format(name)] = torch.sqrt(supp_loss_sq_src).item()
        metrics['{0}/supp_src_dist_rsq_rel'.format(name)] = torch.sqrt(supp_loss_sq_src).item() / vrange

        metrics['{0}/supp_trg_dist_abs'.format(name)] = supp_loss_abs_trg.item()
        metrics['{0}/supp_trg_dist_rel'.format(name)] = supp_loss_abs_trg.item() / vrange
        metrics['{0}/supp_trg_dist_sq'.format(name)] = supp_loss_sq_trg.item()
        metrics['{0}/supp_trg_dist_rsq'.format(name)] = torch.sqrt(supp_loss_sq_trg).item()
        metrics['{0}/supp_trg_dist_rsq_rel'.format(name)] = torch.sqrt(supp_loss_sq_trg).item() / vrange

    return outputs, metrics


def alignment_histplot(src_tensor, trg_tensor, title=None):
    sns.set_style('whitegrid')
    palette = sns.color_palette()
    fig = plt.figure(figsize=(7, 5))

    src_array = src_tensor.numpy()
    trg_array = trg_tensor.numpy()

    mx = max(src_array.max(), trg_array.max())
    mn = min(src_array.min(), trg_array.min())
    d = max(mx - mn, 0.01)
    mn = mn - d * 0.1
    mx = mx + d * 0.1

    sns.histplot(x=src_array, stat='probability', bins=70, binrange=(mn, mx),
                 color=palette[0], label='src', alpha=0.6)
    sns.histplot(x=trg_array, stat='probability', bins=70, binrange=(mn, mx),
                 color=palette[1], label='trg', alpha=0.6)

    plt.ylabel('fraction of samples', fontsize=16)
    plt.xlabel('D(z)', fontsize=16)
    plt.legend(fontsize=16)
    plt.gca().set_xlim(mn, mx)
    ax = plt.gca()
    twin_x = plt.gca().twinx()
    plt.sca(twin_x)
    sns.kdeplot(x=src_array, ax=twin_x, color=palette[0], linewidth=3)
    sns.kdeplot(x=trg_array, ax=twin_x, color=palette[1], linewidth=3)
    plt.gca().set_xlim(mn, mx)
    plt.grid()
    plt.ylabel('')

    tick_labels = ax.xaxis.get_majorticklabels() + ax.yaxis.get_majorticklabels() \
        + twin_x.xaxis.get_majorticklabels() + twin_x.yaxis.get_majorticklabels()
    for label in tick_labels:
        label.set_fontsize(14)

    if title is not None:
        plt.title(title, fontsize=20, y=1.02)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    buf.seek(0)
    image = Image.open(buf)
    np_image = np.array(image.convert("RGB"))
    plt.close(fig)

    return np_image


def extract_features(feature_extractor, loader, device=None):
    was_training = feature_extractor.training
    feature_extractor.eval()

    dim = feature_extractor.n_outputs
    features = torch.empty([0, dim])
    labels = torch.empty([0], dtype=torch.long)
    for batch in loader:
        x, y = batch
        if device is not None:
            x = x.to(device)

        with torch.no_grad():
            z = feature_extractor(x)

            features = torch.cat((features, z.cpu()), dim=0)
            labels = torch.cat((labels, y), dim=0)

    feature_extractor.train(was_training)

    return features, labels


class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def select_points(features, labels, max_num_points=500):
    np.random.seed(1)
    indices = np.random.permutation(np.arange(features.shape[0]))
    n = min(max_num_points, features.shape[0])

    return features[indices[:n]], labels[indices[:n]]


def make_scatter_plot(features_src, labels_src, features_trg, labels_trg, classes,
                      max_num_points=500, src=True, trg=True):
    markers = ['o', 'X', '^', 's']
    palette = sns.color_palette()

    scatter_features_src, scatter_labels_src = select_points(features_src, labels_src,
                                                             max_num_points=max_num_points)
    scatter_features_trg, scatter_labels_trg = select_points(features_trg, labels_trg,
                                                             max_num_points=max_num_points)

    if src:
        for c in classes:
            p_src = scatter_features_src[scatter_labels_src == c]
            plt.scatter(p_src[:, 0], p_src[:, 1], marker=markers[c], color=palette[0], s=45, alpha=0.2,
                        edgecolors='k', label='src (c={})'.format(c))

    if trg:
        for c in classes:
            p_trg = scatter_features_trg[scatter_labels_trg == c]
            plt.scatter(p_trg[:, 0], p_trg[:, 1], marker=markers[c], color=palette[3], s=45, alpha=0.2,
                        edgecolors='k', label='trg (c={})'.format(c))


def make_cls_plot(meshgrid, classifier, device):
    palette = sns.color_palette()
    c_indices = [2, 4, 5, 7]

    X, Y = meshgrid
    M = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    with torch.no_grad():
        M_tensor = torch.tensor(M, dtype=torch.float32).to(device)
        C = classifier(M_tensor).cpu().numpy()

    for i in range(C.shape[1]):
        C_other_max = np.max(np.concatenate((C[:, :i], C[:, i + 1:]), axis=1), axis=1)

        V = C[:, i]
        V = np.ma.masked_array(V, mask=V < C_other_max - 1e-4)

        V = V.reshape(X.shape)

        col = palette[c_indices[i]]
        cmap_f = sns.light_palette(col, reverse=False, as_cmap=True)
        cmap_c = sns.dark_palette(col, reverse=True, as_cmap=True)

        cf = plt.contourf(X, Y, V, levels=10, alpha=0.3, cmap=cmap_f)
        cax = plt.gca().inset_axes([1.06 + 0.2 * i, 0.0, 0.05, 1.0])
        plt.colorbar(cf, ax=plt.gca(), cax=cax)
        plt.contour(X, Y, V, levels=10, alpha=0.6, linewidths=1.0, cmap=cmap_c)


def make_disc_plot(meshgrid, discriminator, device):
    X, Y = meshgrid
    M = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    with torch.no_grad():
        M_tensor = torch.tensor(M, dtype=torch.float32).to(device)
        V = discriminator(M_tensor).cpu().numpy()
    V = V.reshape(X.shape)
    cmap_f = 'seismic'
    cmap_c = sns.color_palette("icefire", as_cmap=True)

    cf = plt.contourf(X, Y, V, levels=10, alpha=0.2, cmap=cmap_f, norm=MidpointNormalize(midpoint=0.0))
    cax = plt.gca().inset_axes([-0.2, 0.0, 0.05, 1.0])
    plt.colorbar(cf, ax=plt.gca(), cax=cax)
    cax.yaxis.tick_left()
    plt.contour(X, Y, V, levels=10, alpha=0.8, linewidths=1, cmap=cmap_c, norm=MidpointNormalize(midpoint=0.0))


def get_lim(array, rel=0.05):
    mx = array.max()
    mn = array.min()
    d = max(mx - mn, 0.01)
    mn = mn - d * rel
    mx = mx + d * rel
    return mn, mx


def feature_plot(algorithm, features_src, labels_src, features_trg, labels_trg, num_classes, device, title=None):

    rows = 2 + num_classes
    fig, axes = plt.subplots(figsize=(14, rows * 5.4), nrows=rows, ncols=2)

    features_src = features_src.numpy()
    labels_src = labels_src.numpy()
    features_trg = features_trg.numpy()
    labels_trg = labels_trg.numpy()

    coords = np.concatenate((features_src, features_trg), axis=0)
    xlim = get_lim(coords[:, 0])
    ylim = get_lim(coords[:, 1])

    x_grid = np.linspace(*xlim, 50)
    y_grid = np.linspace(*ylim, 50)
    meshgrid = np.meshgrid(x_grid, y_grid)

    all_classes = list(range(num_classes))

    plt.sca(axes[0][0])
    if hasattr(algorithm, 'discriminator'):
        make_disc_plot(meshgrid, algorithm.discriminator, device=device)
        make_scatter_plot(features_src, labels_src, features_trg, labels_trg, classes=all_classes)
        plt.title('Discriminator output', fontsize=18, y=1.02)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.sca(axes[0][1])
    make_cls_plot(meshgrid, algorithm.network.classifier, device=device)
    make_scatter_plot(features_src, labels_src, features_trg, labels_trg, classes=all_classes)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Classifier output', fontsize=18, y=1.02)

    plt.sca(axes[1][0])
    make_scatter_plot(features_src, labels_src, features_trg, labels_trg, src=True, trg=False, classes=all_classes)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Source scatter', fontsize=18, y=1.02)

    plt.sca(axes[1][1])
    make_scatter_plot(features_src, labels_src, features_trg, labels_trg, src=False, trg=True, classes=all_classes)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Target scatter', fontsize=18, y=1.02)

    for c in range(num_classes):
        plt.sca(axes[c + 2][0])
        make_scatter_plot(features_src, labels_src, features_trg, labels_trg, src=True, trg=False, classes=[c])
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title('Source scatter (c = {})'.format(c), fontsize=18, y=1.02)

        plt.sca(axes[c + 2][1])
        make_scatter_plot(features_src, labels_src, features_trg, labels_trg, src=False, trg=True, classes=[c])
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title('Target scatter (c = {})'.format(c), fontsize=18, y=1.02)

    fig.legend(*(axes[0][1].get_legend_handles_labels()), fontsize=18, ncol=2,
               loc='lower center', bbox_to_anchor=(0.45, 0.94))

    if title is not None:
        plt.suptitle(title, fontsize=20, y=1.04)

    plt.tight_layout(w_pad=1.2, rect=(0.05, 0.02, 0.95, 0.95))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    np_image = np.array(image.convert("RGB"))
    plt.close(fig)

    return np_image
