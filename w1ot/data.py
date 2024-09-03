import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_swiss_roll, make_s_curve

def generate_gaussian(n_samples, mean, cov):
    return np.random.multivariate_normal(mean, cov, n_samples)

def shuffle_data(array, seed=1):
    indices = np.arange(len(array))
    np.random.seed(seed)
    np.random.shuffle(indices)
    return array[indices]

def make_2d_data(dataset='circles',
                 n_samples=1000,
                 noise=0.01):

    if dataset == '8gaussians' or dataset == '8gaussians_r':
        n_samples_per_cluster = int(n_samples/8)  # 100,000 / 8 for outer clusters
        n_clusters = 8
        radius = 5
        center_cov = [[0.1, 0], [0, 0.1]]
        outer_cov = [[0.1, 0], [0, 0.1]]

        # Generate central (source) clusters
        center_cluster = generate_gaussian(n_samples_per_cluster * 8, [0, 0], center_cov)

        # Generate outer (target) clusters
        outer_clusters = []
        for i in range(n_clusters):
            angle = 2 * np.pi * i / n_clusters
            mean = [radius * np.cos(angle), radius * np.sin(angle)]
            cluster = generate_gaussian(n_samples_per_cluster, mean, outer_cov)
            outer_clusters.append(cluster)

        # Combine all outer clusters
        all_outer_clusters = np.vstack(outer_clusters)

        if dataset == '8gaussians_r':
            return shuffle_data(all_outer_clusters), shuffle_data(center_cluster)
        elif dataset == '8gaussians':
            return shuffle_data(center_cluster), shuffle_data(all_outer_clusters)

    elif dataset == 'checkerboard' or dataset == 'checkerboard_r':
        # Parameters
        num_samples = int(n_samples/9)  # Number of samples per distribution
        cov = [[0.1, 0], [0, 0.1]]  # Covariance matrix (small spread)
        k = 3
        # Define the centers of the 5 source and 4 target Gaussian distributions
        source_means = [
            [0, 0], [k, k], [-k, k], [k, -k], [-k, -k]
        ]
        target_means = [
            [k, 0], [0, k], [-k, 0], [0, -k]
        ]

        # Generate source data
        source_data = [generate_gaussian(num_samples, mean, cov) for mean in source_means]
        # Generate target data
        target_data = [generate_gaussian(num_samples, mean, cov) for mean in target_means]

        # transform to the numpy array
        source_data = np.vstack(source_data)
        target_data = np.vstack(target_data)

        if dataset == 'checkerboard_r':
            return shuffle_data(target_data), shuffle_data(source_data)
        elif dataset == 'checkerboard':
            return shuffle_data(source_data), shuffle_data(target_data)

    elif dataset == 'moons' or dataset == 'moons_r':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=1)
        source = X[y == 0]
        target = X[y == 1]
        if dataset == 'moon_r':
            return shuffle_data(target), shuffle_data(source)
        elif dataset == 'moons':
            return shuffle_data(source), shuffle_data(target)

    elif dataset == 'circles' or dataset == 'circles_r':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
        source = X[y == 1]
        target = X[y == 0]
        if dataset == 'circles_r':
            return shuffle_data(target), shuffle_data(source)
        elif dataset == 'circles':
            return shuffle_data(source), shuffle_data(target)

    elif dataset == 'swiss_roll':
        source = generate_gaussian(n_samples, [0, 0], [[1, 0], [0, 1]])/2
        X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
        target = X[:, [0, 2]]/4
        return shuffle_data(source), shuffle_data(target)

    elif dataset == 's_curve':
        source = generate_gaussian(n_samples, [0, 0], [[0.05, 0], [0, 0.05]])
        X, _ = make_s_curve(n_samples=n_samples, noise=noise)
        target = X[:, [2, 0]]
        return shuffle_data(source), shuffle_data(target)


def plot_2d_data(source, target, transported=None, transport_ray_size=1, title=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(source[:, 0], source[:, 1], alpha=0.5, s=1, c='b', label='Source')
    plt.scatter(target[:, 0], target[:, 1],  alpha=0.5, s=1, c='r', label='Target')
    if transported is not None:
        plt.scatter(transported[:, 0], transported[:, 1], alpha=0.5, label='Transported', color='green', s=1)
        num_rays = int(len(source)*transport_ray_size)  # Limit the number of rays to avoid clutter
        indices = np.random.choice(len(source), num_rays, replace=False)

        for i in indices:
            plt.plot([source[i, 0], transported[i, 0]],
                     [source[i, 1], transported[i, 1]],
                     color='gray', alpha=0.3, linewidth=0.5)
    plt.legend(loc='upper right')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.tight_layout()
    plt.show()

import anndata
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


## below is the data reading code borrowed from cellot to follow the same data split and configuration
## cellot: http://localhost:2023/lab/tree/Desktop/cellot/cellot/data/cell.py






from absl import flags
from ml_collections import ConfigDict
import yaml
import re
import string

ALPHABET = string.ascii_lowercase + string.digits

flags.DEFINE_string("outroot", "./results", "Root directory to write model output")

flags.DEFINE_string("model_name", "", "Name of model class")

flags.DEFINE_string("data_name", "", "Name of dataset")

flags.DEFINE_string("preproc_name", "", "Name of dataset")

flags.DEFINE_string("experiment_name", "", "Name for experiment")

flags.DEFINE_string("submission_id", "", "UUID generated by bash script submitting job")

flags.DEFINE_string(
    "drug", "", "Compute OT map on drug, change outdir to outdir/drugs/drug"
)

flags.DEFINE_string("celldata", "", "Short cut to specify config.data.path & outdir")

flags.DEFINE_string("outdir", "", "Path to outdir")

FLAGS = flags.FLAGS

def load_config(path, unparsed=None):

    path = Path(path)

    if path.exists():
        config = ConfigDict(yaml.load(open(path, "r"), yaml.SafeLoader))
    else:
        print("WARNING: config path not found")
        config = ConfigDict()

    if unparsed is not None:
        opts = parse_cli_opts(unparsed)
        config.update(opts)

    return config

def parse_cli_opts(args):
    def parse(argiter):
        for opt in argiter:
            if "=" not in opt:
                value = next(argiter)
                key = re.match(r"^--config\.(?P<key>.*)", opt)["key"]

            else:
                match = re.match(r"^--config\.(?P<key>.*)=(?P<value>.*)", opt)
                key, value = match["key"], match["value"]

            value = yaml.load(value, Loader=yaml.UnsafeLoader)
            yield key, value

    opts = dict()
    if len(args) == 0:
        return opts

    argiter = iter(args)
    for key, val in parse(argiter):
        *tree, leaf = key.split(".")
        lut = opts
        for k in tree:
            lut = lut.setdefault(k, dict())
        lut[leaf] = val

    return opts

def parse_config_cli(path, args):
    if isinstance(path, list):
        config = ConfigDict()
        for path in FLAGS.config:
            config.update(yaml.load(open(path), Loader=yaml.UnsafeLoader))
    else:
        config = load_config(path)

    opts = parse_cli_opts(args)
    config.update(opts)

    if len(FLAGS.celldata) > 0:
        config.data.path = str(FLAGS.celldata)
        config.data.type = "cell"
        config.data.source = "control"

    drug = FLAGS.drug
    if len(drug) > 0:
        config.data.target = drug

    return config

def read_list(arg):

    if isinstance(arg, str):
        arg = Path(arg)
        assert arg.exists()
        lst = arg.read_text().split()
    else:
        lst = arg

    return list(lst)

def read_single_anndata(config, path=None):
    if path is None:
        path = config.data.path

    data = anndata.read(path)

    if "features" in config.data:
        features = read_list(config.data.features)
        data = data[:, features].copy()

    # select subgroup of individuals
    if "individuals" in config.data:
        data = data[
            data.obs[config.data.individuals[0]].isin(config.data.individuals[1])
        ]

    # label conditions as source/target distributions
    # config.data.{source,target} can be a list now
    transport_mapper = dict()
    for value in ["source", "target"]:
        key = config.data[value]
        if isinstance(key, list):
            for item in key:
                transport_mapper[item] = value
        else:
            transport_mapper[key] = value

    data.obs["transport"] = data.obs[config.data.condition].apply(transport_mapper.get)

    if config.data["target"] == "all":
        data.obs["transport"].fillna("target", inplace=True)

    mask = data.obs["transport"].notna()
    assert "subset" not in config.data
    if "subset" in config.datasplit:
        for key, value in config.datasplit.subset.items():
            if not isinstance(value, list):
                value = [value]
            mask = mask & data.obs[key].isin(value)

    # write train/test/valid into split column
    data = data[mask].copy()
    if "datasplit" in config:
        data.obs["split"] = split_cell_data(data, **config.datasplit)

    return data

def split_cell_data_train_test(
    data, groupby=None, random_state=0, holdout=None, subset=None, **kwargs
):

    split = pd.Series(None, index=data.obs.index, dtype=object)
    groups = {None: data.obs.index}
    if groupby is not None:
        groups = data.obs.groupby(groupby).groups

    for key, index in groups.items():
        trainobs, testobs = train_test_split(index, random_state=random_state, **kwargs)
        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = "ood"

    return split


def split_cell_data_train_test_eval(
    data,
    test_size=0.15,
    eval_size=0.15,
    groupby=None,
    random_state=0,
    holdout=None,
    **kwargs
):

    split = pd.Series(None, index=data.obs.index, dtype=object)

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = "ood"

    groups = {None: data.obs.loc[split != "ood"].index}
    if groupby is not None:
        groups = data.obs.loc[split != "ood"].groupby(groupby).groups

    for key, index in groups.items():
        training, evalobs = train_test_split(
            index, random_state=random_state, test_size=eval_size
        )

        trainobs, testobs = train_test_split(
            training, random_state=random_state, test_size=test_size
        )

        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"
        split.loc[evalobs] = "eval"

    return split


def split_cell_data_toggle_ood(data, holdout, key, mode, random_state=0, **kwargs):

    """Hold out ood sample, coordinated with iid split

    ood sample defined with key, value pair

    for ood mode: hold out all cells from a sample
    for iid mode: include half of cells in split
    """

    split = split_cell_data_train_test(data, random_state=random_state, **kwargs)

    if not isinstance(holdout, list):
        value = [holdout]

    ood = data.obs_names[data.obs[key].isin(value)]
    trainobs, testobs = train_test_split(ood, random_state=random_state, test_size=0.5)

    if mode == "ood":
        split.loc[trainobs] = "ignore"
        split.loc[testobs] = "ood"

    elif mode == "iid":
        split.loc[trainobs] = "train"
        split.loc[testobs] = "ood"

    else:
        raise ValueError

    return split


def split_cell_data(data, name="train_test", **kwargs):
    if name == "train_test":
        split = split_cell_data_train_test(data, **kwargs)
    elif name == "toggle_ood":
        split = split_cell_data_toggle_ood(data, **kwargs)
    elif name == "train_test_eval":
        split = split_cell_data_train_test_eval(data, **kwargs)
    else:
        raise ValueError

    return split.astype("category")



