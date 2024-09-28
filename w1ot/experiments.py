from w1ot.ot import w1ot, w2ot
from w1ot.vae import VAE
from w1ot.utils import ensure_numpy
import os
import anndata as ad
import pandas as pd
import pickle
import numpy as np
from typing import Any, Dict
from anndata import AnnData
import ray
import torch

class Experiment:
    def __init__(self, 
                 model_name: str, 
                 dataset: AnnData, 
                 embedding: bool,
                 latent_dim: int,
                 output_dir: str,
                 test_size: float = 0.1,
                 device: str = None,
                 num_cpus: int = 4,
                 num_gpus: int = 0,
                 ray_address: str = None,
                 num_runs: int = 1) -> None:
        """
        Initialize the Experiment class.

        Parameters:
            model_name (str): Name of the model to use. Options are 'w1ot', 'w2ot'.
            dataset (AnnData): The anndata dataset containing observations with annotations.
            embedding (bool): Whether to use the embedding or not.
            latent_dim (int): Dimension of the latent space for the embedding.
            output_dir (str): Directory where results and models will be saved.
            test_size (float, optional): Proportion of the dataset to use for testing. Defaults to 0.1.
            device (str, optional): Computing device to use. If None, uses CUDA if available, else CPU. Defaults to None.
            num_cpus (int, optional): Number of CPUs to allocate for Ray tasks. Defaults to 4.
            num_gpus (int, optional): Number of GPUs to allocate for Ray tasks. Defaults to 0.
            num_runs (int, optional): Number of times to repeat the experiment. Defaults to 1.

        Example:
            >>> import anndata
            >>> adata = anndata.read_h5ad("my_dataset.h5ad")
            >>> experiment = Experiment(model_name='w1ot',
            ...                         dataset=adata,
            ...                         embedding=True,
            ...                         latent_dim=50,
            ...                         output_dir='./results',
            ...                         test_size=0.1,
            ...                         device='cuda',
            ...                         num_cpus=8,
            ...                         num_gpus=1,
            ...                         num_runs=3)
        """
        self.model_name = model_name
        if model_name == 'w1ot':
            self.model = w1ot
        elif model_name == 'w2ot':
            self.model = w2ot
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results: Dict[str, Dict[str, Any]] = {}
        self.test_size = test_size
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = embedding
        self.latent_dim = latent_dim

        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.num_runs = num_runs

        # Initialize Ray
        if not ray.is_initialized():
            if ray_address:
                # Connect to an existing Ray cluster
                ray.init(address=ray_address)
            else:
                # Start a local Ray instance
                ray.init()

    def run_iid_perturbation(self) -> None:
        """
        Run IID perturbation experiments in parallel using Ray.

        This method divides the IID perturbation experiments into subtasks, each processing a unique perturbation.
        The subtasks are executed in parallel to accelerate the experiment.

        Example:
            >>> experiment.run_iid_perturbation()
        """
        perturbations = self.dataset.obs['perturbation'].unique()
        perturbations = [p for p in perturbations if p != 'control']

        # Put shared data into Ray's object store to minimize data movement
        dataset_ref = ray.put(self.dataset)

        for run in range(1, self.num_runs + 1):
            run_dir = os.path.join(self.output_dir, f'{self.model_name}_run{run}')
            os.makedirs(run_dir, exist_ok=True)

            # Create tasks to process each perturbation in parallel
            tasks = [process_iid_perturbation.options(
                    num_cpus=self.num_cpus,
                    num_gpus=self.num_gpus).remote(
                    model_name=self.model_name,
                    dataset=dataset_ref,
                    perturbation=perturbation,
                    embedding=self.embedding,
                    latent_dim=self.latent_dim,
                    model_dir=run_dir,
                    test_size=self.test_size,
                    device=self.device
                 ) for perturbation in perturbations]
            # Wait for all tasks to complete
            ray.get(tasks)

    def run_out_of_distribution(self, ood_type: str, train_category: str) -> None:
        """
        Run out-of-distribution perturbation experiments in parallel using Ray.

        This method divides the out-of-distribution experiments into subtasks, each processing a unique perturbation.
        The subtasks are executed in parallel to accelerate the experiment.

        Parameters:
            ood_type (str): Type of out-of-distribution experiment to run. Options are 'dosage', 'celltype'.
            train_category (str): The category to use for training.

        Example:
            >>> experiment.run_out_of_distribution(ood_type='celltype', train_category='B_cell')
        """
        if ood_type == 'dosage':
            categories = self.dataset.obs['dosage'].unique()
        elif ood_type == 'celltype':
            categories = self.dataset.obs['celltype'].unique()
        else:
            raise ValueError(f"Unknown type: {ood_type}")

        perturbations = self.dataset.obs['perturbation'].unique()
        # Exclude 'control' in the perturbation list
        perturbations = [perturbation for perturbation in perturbations if perturbation != 'control']

        # Put shared data into Ray's object store to minimize data movement
        dataset_ref = ray.put(self.dataset)

        for run in range(1, self.num_runs + 1):
            run_dir = os.path.join(self.output_dir, f'{self.model_name}_run{run}')
            os.makedirs(run_dir, exist_ok=True)

            # Create tasks to process each perturbation in parallel
            tasks = [process_ood_perturbation.options(
                        num_cpus=self.num_cpus,
                        num_gpus=self.num_gpus).remote(
                        model_name=self.model_name,
                        dataset=dataset_ref,
                        ood_type=ood_type,
                        train_category=train_category,
                        perturbation=perturbation,
                        categories=categories,
                        embedding=self.embedding,
                        latent_dim=self.latent_dim,
                        model_dir=run_dir,
                        device=self.device
                     ) for perturbation in perturbations]

            # Wait for all tasks to complete
            ray.get(tasks)

    def shutdown(self) -> None:
        """
        Shutdown the Ray instance.

        Example:
            >>> experiment.shutdown()
        """
        if ray.is_initialized():
            ray.shutdown()

@ray.remote(num_cpus=4, num_gpus=1)
def process_iid_perturbation(model_name: str,
                             dataset: AnnData,
                             perturbation: str,
                             embedding: bool,
                             latent_dim: int,
                             model_dir: str,
                             test_size: float,
                             device: str) -> None:
    """
    Process a single IID perturbation in parallel.

    This function runs in a separate process managed by Ray.

    Parameters:
        model_name (str): Name of the model to use ('w1ot' or 'w2ot').
        dataset_ref (ray.ObjectRef): Reference to the dataset stored in Ray's object store.
        perturbation (str): The perturbation to process.
        embedding (bool): Whether to use embedding.
        latent_dim (int): Dimensionality of the latent space.
        model_dir (str): Directory to save models and results.
        test_size (float): Proportion of the dataset to use for testing.
        device (str): Computing device to use.

    Example:
        >>> process_iid_perturbation.remote(model_name='w1ot',
        ...                                  dataset=dataset_ref,
        ...                                  perturbation='drugA',
        ...                                  embedding=True,
        ...                                  latent_dim=50,
        ...                                  model_dir='./results/w1ot',
        ...                                  test_size=0.1,
        ...                                  device='cuda')
    """
    import os
    import torch
    import anndata as ad
    from anndata import AnnData
    import numpy as np
    from w1ot.ot import w1ot, w2ot
    from w1ot.vae import VAE
    from w1ot.utils import ensure_numpy

    if model_name == 'w1ot':
        model_cls = w1ot
    elif model_name == 'w2ot':
        model_cls = w2ot
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    source_adata = dataset[dataset.obs['perturbation'] == 'control'].copy()
    target_adata = dataset[dataset.obs['perturbation'] == perturbation].copy()

    # Create directories for saving models and results
    model_path = os.path.join(model_dir, f'{perturbation}_{latent_dim}')
    os.makedirs(model_path, exist_ok=True)

    # Split the data into train and test sets
    source_test_size = int(len(source_adata) * test_size)
    target_test_size = int(len(target_adata) * test_size)
    source_train_adata = source_adata[:-source_test_size]
    source_test_adata = source_adata[-source_test_size:]
    target_train_adata = target_adata[:-target_test_size]
    target_test_adata = target_adata[-target_test_size:]

    # Embed the data if required
    if embedding:
        # Initialize and train the VAE model
        embedding_model = VAE(device=device, output_dir=model_path)
        embedding_model.setup_anndata(ad.concat([source_train_adata, target_train_adata]))
        embedding_model.setup_model(hidden_layers=[128, 128], latent_dim=latent_dim)
        embedding_model.train(num_iters=10000, batch_size=256, lr=1e-4,
                              resume_from_checkpoint=True, checkpoint_interval=100)

        # Get embeddings
        source_train = embedding_model.get_latent_representation(source_train_adata)
        source_train_adata.obsm['X_emb'] = source_train
        target_train = embedding_model.get_latent_representation(target_train_adata)
        target_train_adata.obsm['X_emb'] = target_train
        source_test = embedding_model.get_latent_representation(source_test_adata)
        source_test_adata.obsm['X_emb'] = source_test
        target_test = embedding_model.get_latent_representation(target_test_adata)
        target_test_adata.obsm['X_emb'] = target_test
    else:
        source_train = ensure_numpy(source_train_adata.X)
        target_train = ensure_numpy(target_train_adata.X)
        source_test = ensure_numpy(source_test_adata.X)
        target_test = ensure_numpy(target_test_adata.X)

    # Initialize and train the OT model
    model = model_cls(source=source_train, target=target_train, device=device, path=model_path)
    model.fit_potential_function()
    if model_name == 'w1ot':
        model.fit_distance_function()

    model.save(model_path)

    # Predict on test data
    transported = model.transport(source_test)

    # Decode the transported data if embedding was used
    if embedding:
        reconstructed = embedding_model.decode(transported)
        transported_adata = AnnData(X=reconstructed, obs=source_test_adata.obs, var=source_test_adata.var)
        transported_adata.obsm['X_emb'] = transported
    else:
        transported_adata = AnnData(X=transported, obs=source_test_adata.obs, var=source_test_adata.var)

    # Save results
    source_train_adata.write(os.path.join(model_path, f'source_train_{perturbation}.h5ad'))
    target_train_adata.write(os.path.join(model_path, f'target_train_{perturbation}.h5ad'))
    source_test_adata.write(os.path.join(model_path, f'source_test_{perturbation}.h5ad'))
    target_test_adata.write(os.path.join(model_path, f'target_test_{perturbation}.h5ad'))
    transported_adata.write(os.path.join(model_path, f'transported_{perturbation}.h5ad'))

@ray.remote(num_cpus=4, num_gpus=1)
def process_ood_perturbation(model_name: str,
                             dataset: AnnData,
                             ood_type: str,
                             train_category: str,
                             perturbation: str,
                             categories: list,
                             embedding: bool,
                             latent_dim: int,
                             model_dir: str,
                             device: str) -> None:
    """
    Process a single out-of-distribution perturbation experiment in parallel.

    Parameters:
        model_name (str): Name of the model to use ('w1ot' or 'w2ot').
        dataset_ref (ray.ObjectRef): Reference to the dataset stored in Ray's object store.
        ood_type (str): Type of out-of-distribution experiment ('dosage' or 'celltype').
        train_category (str): The category to use for training.
        perturbation (str): The perturbation to process.
        categories (list): List of all categories in the ood_type.
        embedding (bool): Whether to use embedding.
        latent_dim (int): Dimensionality of the latent space.
        model_dir (str): Directory to save models and results.
        device (str): Computing device to use.

    Example:
        >>> process_ood_perturbation.remote(model_name='w1ot',
        ...                                  dataset_ref=dataset_ref,
        ...                                  ood_type='celltype',
        ...                                  train_category='B_cell',
        ...                                  perturbation='drugB',
        ...                                  categories=['B_cell', 'T_cell', 'Monocyte'],
        ...                                  embedding=True,
        ...                                  latent_dim=50,
        ...                                  model_dir='./results/w1ot/celltype',
        ...                                  device='cuda')
    """
    import os
    import torch
    import anndata as ad
    from anndata import AnnData
    import numpy as np
    from w1ot.ot import w1ot, w2ot
    from w1ot.vae import VAE
    from w1ot.utils import ensure_numpy

    # Retrieve the dataset from Ray's object store
    dataset = ray.get(dataset_ref)

    if model_name == 'w1ot':
        model_cls = w1ot
    elif model_name == 'w2ot':
        model_cls = w2ot
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Train data: control and perturbed data for the training category
    source_train_adata = dataset[
        (dataset.obs[ood_type] == train_category) &
        (dataset.obs['perturbation'] == 'control')
    ].copy()

    target_train_adata = dataset[
        (dataset.obs[ood_type] == train_category) &
        (dataset.obs['perturbation'] == perturbation)
    ].copy()

    # Create directories for saving models and results
    ood_type_dir = os.path.join(model_dir, ood_type)
    os.makedirs(ood_type_dir, exist_ok=True)

    model_path = os.path.join(ood_type_dir, f'{perturbation}_{latent_dim}_{train_category}')
    os.makedirs(model_path, exist_ok=True)

    # Embed the training data if required
    if embedding:
        # Initialize and train the VAE model
        embedding_model = VAE(device=device, output_dir=model_path)
        embedding_model.setup_anndata(ad.concat([source_train_adata, target_train_adata]))
        embedding_model.setup_model(hidden_layers=[128, 128], latent_dim=latent_dim)
        embedding_model.train(num_iters=10000, batch_size=256, lr=1e-4,
                              resume_from_checkpoint=True, checkpoint_interval=100)

        # Get embeddings
        source_train = embedding_model.get_latent_representation(source_train_adata)
        source_train_adata.obsm['X_emb'] = source_train
        target_train = embedding_model.get_latent_representation(target_train_adata)
        target_train_adata.obsm['X_emb'] = target_train
    else:
        source_train = ensure_numpy(source_train_adata.X)
        target_train = ensure_numpy(target_train_adata.X)

    # Initialize and train the OT model
    model = model_cls(source=source_train, target=target_train, device=device, path=model_path)
    model.fit_potential_function()
    if model_name == 'w1ot':
        model.fit_distance_function()

    # Save the trained model
    model.save(model_path)

    # Test on all other categories
    for test_category in categories:
        if test_category == train_category:
            continue

        source_test_adata = dataset[
            (dataset.obs[ood_type] == test_category) &
            (dataset.obs['perturbation'] == 'control')
        ].copy()

        target_test_adata = dataset[
            (dataset.obs[ood_type] == test_category) &
            (dataset.obs['perturbation'] == perturbation)
        ].copy()

        # Embed the test data if required
        if embedding:
            source_test = embedding_model.get_latent_representation(source_test_adata)
            source_test_adata.obsm['X_emb'] = source_test
            target_test = embedding_model.get_latent_representation(target_test_adata)
            target_test_adata.obsm['X_emb'] = target_test
        else:
            source_test = ensure_numpy(source_test_adata.X)
            target_test = ensure_numpy(target_test_adata.X)

        # Predict on test data
        transported = model.transport(source_test)
        
        # Transform the transported data back to the original space if embedding was used
        if embedding:
            reconstructed = embedding_model.decode(transported)
            transported_adata = AnnData(X=reconstructed, obs=source_test_adata.obs, var=source_test_adata.var)
            transported_adata.obsm['X_emb'] = transported
        else:
            transported_adata = AnnData(X=transported, obs=source_test_adata.obs, var=source_test_adata.var)

        # Save results as AnnData files
        source_test_adata.write(os.path.join(
            model_path, f'source_test_{ood_type}_{train_category}_{test_category}_{perturbation}.h5ad'))
        target_test_adata.write(os.path.join(
            model_path, f'target_test_{ood_type}_{train_category}_{test_category}_{perturbation}.h5ad'))
        transported_adata.write(os.path.join(
            model_path, f'transported_{ood_type}_{train_category}_{test_category}_{perturbation}.h5ad'))

    # Save training data
    source_train_adata.write(os.path.join(
        model_path, f'source_train_{ood_type}_{train_category}_{perturbation}.h5ad'))
    target_train_adata.write(os.path.join(
        model_path, f'target_train_{ood_type}_{train_category}_{perturbation}.h5ad'))
