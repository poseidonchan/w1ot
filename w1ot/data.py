import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_swiss_roll, make_s_curve

def generate_gaussian(n_samples, mean, cov):
    """
    Generates a Gaussian-distributed dataset.

    Parameters:
        n_samples (int): Number of samples to generate.
        mean (list or array-like): Mean of the Gaussian distribution.
        cov (list or array-like): Covariance matrix of the Gaussian distribution.

    Returns:
        np.ndarray: Generated data samples with shape (n_samples, 2).
    """
    return np.random.multivariate_normal(mean, cov, n_samples)

def shuffle_data(array, seed=1):
    """
    Shuffles the data array in a deterministic manner based on the provided seed.

    Parameters:
        array (np.ndarray): Array of data to be shuffled.
        seed (int, optional): Seed for the random number generator. Defaults to 1.

    Returns:
        np.ndarray: Shuffled data array.
    """
    indices = np.arange(len(array))
    np.random.seed(seed)
    np.random.shuffle(indices)
    return array[indices]

def make_2d_data(dataset='circles',
                n_samples=1000,
                noise=0.01):
    """
    Generates 2D datasets based on the specified type.

    Parameters:
        dataset (str, optional): Type of dataset to generate. Options include:
                                 '8gaussians', '8gaussians_r',
                                 'checkerboard', 'checkerboard_r',
                                 'bookshelf', 'moons', 'moons_r',
                                 'circles', 'circles_r', 'swiss_roll', 's_curve'.
                                 Defaults to 'circles'.
        n_samples (int, optional): Number of samples to generate. Defaults to 1000.
        noise (float, optional): Standard deviation of Gaussian noise added to the data. Defaults to 0.01.

    Returns:
        tuple: A tuple containing source and target datasets as numpy arrays.
    """
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

        # Transform to the numpy array
        source_data = np.vstack(source_data)
        target_data = np.vstack(target_data)

        if dataset == 'checkerboard_r':
            return shuffle_data(target_data), shuffle_data(source_data)
        elif dataset == 'checkerboard':
            return shuffle_data(source_data), shuffle_data(target_data)

    elif dataset == 'bookshelf':
        source = np.hstack((np.random.uniform(0, 1, (n_samples, 1)), np.random.normal(0, 0.001, (n_samples, 1))))
        target = np.hstack((np.random.uniform(2, 3, (n_samples, 1)), np.random.normal(0, 0.001, (n_samples, 1))))
        
        return shuffle_data(source), shuffle_data(target)

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

        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
        source = np.concatenate((X[y==1]*3, source))
        target = np.concatenate((X[y==0]*2, target))

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


def plot_2d_data(source, target, transported=None, show_relative_position=False, transport_ray_ratio=1, title=None):
    """
    Visualizes 2D source and target datasets with optional transported data.

    This function creates a scatter plot of the source and target data points in a 2D space.
    Optionally, it can also plot transported data points and visualize the transport directions
    between source and transported points. Additionally, it can highlight the relative positions
    of selected points for better comparison.

    Parameters:
        source (np.ndarray): Array of source data points with shape (n_samples, 2).
        target (np.ndarray): Array of target data points with shape (n_samples, 2).
        transported (np.ndarray, optional): Array of transported data points with shape (n_samples, 2). Defaults to None.
        show_relative_position (bool, optional): If True, highlights the relative positions of randomly selected points. Defaults to False.
        transport_ray_ratio (float, optional): Ratio to determine the number of transport rays for visualization. Defaults to 1.
        title (str, optional): Title of the plot. Defaults to None.

    Returns:
        None
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(source[:, 0], source[:, 1], alpha=0.5, s=1, c='b', label='Source')
    plt.scatter(target[:, 0], target[:, 1],  alpha=0.5, s=1, c='r', label='Target')
    
    if transported is not None:
        plt.scatter(transported[:, 0], transported[:, 1], alpha=0.5, label='Transported', color='green', s=1)
        num_rays = int(len(source) * transport_ray_ratio)  # Limit the number of rays to avoid clutter
        indices = np.random.choice(len(source), num_rays, replace=False)

        for i in indices:
            plt.plot([source[i, 0], transported[i, 0]],
                     [source[i, 1], transported[i, 1]],
                     color='gray', alpha=0.1, linewidth=0.5)

        if show_relative_position:
            # Randomly pick 5 points from source
            source_indices = np.random.choice(len(source), 5, replace=False)
            markers = ['^', 'o', 's', 'D', 'x']  # Triangle, square, circle, diamond, cross
            colors = ['orange', 'purple']  # Colors for source and transported

            for i, idx in enumerate(source_indices):
                plt.scatter(source[idx, 0], source[idx, 1], marker=markers[i], color=colors[0], s=100, label='Selected Source' if i == 0 else "")
                plt.scatter(transported[idx, 0], transported[idx, 1], marker=markers[i], color=colors[1], s=100, label='Selected Transported' if i == 0 else "")

    # Set the x and y axis range
    plt.xlim(min(source[:, 0].min(), target[:, 0].min()) - 0.1, max(source[:, 0].max(), target[:, 0].max()) + 0.1)
    plt.ylim(min(source[:, 1].min(), target[:, 1].min()) - 0.1, max(source[:, 1].max(), target[:, 1].max()) + 0.1)
    plt.legend(loc='upper right')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.tight_layout()
    plt.show()


