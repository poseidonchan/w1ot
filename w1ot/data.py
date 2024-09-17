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
        n_samples_per_cluster = int(n_samples/8)
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
        num_samples = int(n_samples / 2)  # Number of samples per distribution
        source_samples = int(num_samples / 5)
        target_samples = int(num_samples / 4)
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
        source_data = [generate_gaussian(source_samples, mean, cov) for mean in source_means]
        # Generate target data
        target_data = [generate_gaussian(target_samples, mean, cov) for mean in target_means]

        # transform to the numpy array
        source_data = np.vstack(source_data)
        target_data = np.vstack(target_data)

        if dataset == 'checkerboard_r':
            return shuffle_data(target_data), shuffle_data(source_data)
        elif dataset == 'checkerboard':
            return shuffle_data(source_data), shuffle_data(target_data)

    elif dataset == 'moons' or dataset == 'moons_r':
        X, y = make_moons(n_samples=n_samples, noise=noise)
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





