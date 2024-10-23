import numpy as np
import anndata as ad
from anndata import AnnData
import scanpy as sc
from w1ot import w1ot, w2ot
from w1ot.vae import VAE
from w1ot.eval import metrics
from w1ot.utils import normalize_and_log_transform, ensure_numpy
from typing import List

class PerturbModel:
    def __init__(self,
                 model_name: str,
                 source_adata: AnnData,
                 target_adata: AnnData,
                 perturbation_attribute: str = 'perturbation',
                 embedding: bool = True,
                 latent_dim: int = 50,
                 output_dir: str = None,
                 hidden_layers: List[int] = [512, 512],
                 num_iters: int = 250000,
                 device: str = None) -> None:

        self.model_name = model_name
        self.source_adata = source_adata
        self.target_adata = target_adata
        self.latent_dim = latent_dim
        self.output_dir = output_dir
        self.device = device
        self.embedding = embedding
        self.perturbation_attribute = perturbation_attribute
        self.hidden_layers = hidden_layers
        self.num_iters = num_iters

    def train(self) -> None:
        if self.embedding:
            self.embedding_model = VAE(alpha=0.0, device=self.device, output_dir=self.output_dir)
            # balance the source and target data in the embedding space
            vae_sample_size = min(self.source_adata.shape[0], self.target_adata.shape[0])
            source_ind = np.random.choice(range(self.source_adata.shape[0]), size=vae_sample_size, replace=False)
            target_ind = np.random.choice(range(self.target_adata.shape[0]), size=vae_sample_size, replace=False)
            source_adata_vae = self.source_adata[source_ind, :].copy()
            target_adata_vae = self.target_adata[target_ind, :].copy()
            self.embedding_model.setup_anndata(ad.concat([source_adata_vae, target_adata_vae]))
            self.embedding_model.setup_model(hidden_layers=self.hidden_layers, latent_dim=self.latent_dim)

            if 'scgen' in self.model_name:
                self.embedding_model.train(num_iters=self.num_iters, resume_from_checkpoint=True, checkpoint_interval=2500)
            else:
                scgen_output_dir = os.path.join(self.output_dir, '../scgen')
                if os.path.exists(os.path.join(scgen_output_dir, 'vae_checkpoint.pth')):
                    self.embedding_model.load(os.path.join(scgen_output_dir, 'vae_checkpoint.pth'))
                else:
                    self.embedding_model.train(num_iters=self.num_iters, resume_from_checkpoint=True, checkpoint_interval=2500)

            source = self.embedding_model.get_latent_representation(self.source_adata)
            target = self.embedding_model.get_latent_representation(self.target_adata) 
        
        else:
            source = normalize_and_log_transform(ensure_numpy(self.source_adata.X))
            target = normalize_and_log_transform(ensure_numpy(self.target_adata.X))
        
        print(source.shape, target.shape, source.max(), target.max(), source.min(), target.min())
    
        if self.model_name == 'w1ot':
            self.model = w1ot(
                source=source, 
                target=target,
                device=self.device, 
                path=self.output_dir
            )
            try:
                self.model.load(os.path.join(self.output_dir, "w1ot_networks.pt"))
                print("Loaded existing trained model")
            except:
                self.model.fit_potential_function(resume_from_checkpoint=True)
                self.model.fit_distance_function(resume_from_checkpoint=True)

        elif self.model_name == 'w2ot':
            self.model = w2ot(
                source=source, 
                target=target, 
                device=self.device, 
                path=self.output_dir
            )
            try:
                self.model.load(os.path.join(self.output_dir, "w2ot_networks.pt"))
                print("Loaded existing trained model")
            except:
                self.model.fit_potential_function(resume_from_checkpoint=True)
            
        elif self.model_name == 'scgen':
            eq = min(self.source_adata.X.shape[0], self.target_adata.X.shape[0])
            source_ind = np.random.choice(range(self.source_adata.shape[0]), size=eq, replace=False)
            target_ind = np.random.choice(range(self.target_adata.shape[0]), size=eq, replace=False)
            source_adata_sampled = self.source_adata[source_ind, :]
            target_adata_sampled = self.target_adata[target_ind, :]
            if self.embedding:
                latent_source = np.mean(
                    self.embedding_model.get_latent_representation(source_adata_sampled), 
                    axis=0
                )
                latent_target = np.mean(
                    self.embedding_model.get_latent_representation(target_adata_sampled), 
                    axis=0
                )
            else:
                latent_source = np.mean(normalize_and_log_transform(ensure_numpy(source_adata_sampled.X)), axis=0)
                latent_target = np.mean(normalize_and_log_transform(ensure_numpy(target_adata_sampled.X)), axis=0)
            
            self.delta = latent_target - latent_source

        else:
            raise ValueError(f"Unknown model name: {self.model_name}")


    def predict(self, 
                source_adata: AnnData,
                ) -> AnnData:
        if self.embedding:
            source = self.embedding_model.get_latent_representation(source_adata)
        else:
            source = normalize_and_log_transform(ensure_numpy(source_adata.X))

        if self.model_name in ['w1ot', 'w2ot']:
            transported = self.model.transport(source)
        elif self.model_name == 'scgen':    
            transported = source + self.delta
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        if self.embedding:
            transported_cells = self.embedding_model.decode(transported)
        else:
            transported_cells = np.clip(transported, 0, np.inf)

        transported_adata = AnnData(
            X=transported_cells,
            obs=source_adata.obs.copy(),
            var=source_adata.var.copy(),
        )
        if self.embedding:
            transported_adata.obsm['X_emb'] = transported

        return transported_adata
    
    def evaluate(self, source_adata, target_adata, top_k=50):
        # balance the source and target data when evaluating
        eval_sample_size = min(source_adata.shape[0], target_adata.shape[0])
        source_ind = np.random.choice(range(source_adata.shape[0]), size=eval_sample_size, replace=False)
        target_ind = np.random.choice(range(target_adata.shape[0]), size=eval_sample_size, replace=False)
        source_adata = source_adata[source_ind, :].copy()
        target_adata = target_adata[target_ind, :].copy()

        if top_k > target_adata.shape[1]:
            top_k = target_adata.shape[1]
            gene_list = None
        else:
            try:
                perturbation = target_adata.obs[self.perturbation_attribute].unique()[0]
                gene_list = target_adata.varm['marker_genes-drug-rank'][perturbation].sort_values()[:top_k].index
                print("Loaded existing gene list")
            except:
                adata = ad.concat([source_adata, target_adata])
                sc.tl.rank_genes_groups(adata, self.perturbation_attribute, reference='control')
                perturbation = target_adata.obs[self.perturbation_attribute].unique()[0]
                gene_list = adata.uns['rank_genes_groups']['names'][perturbation][:top_k]
                print("Created new gene list")
        
        if self.model_name == 'identity':
            cell_r2, cell_l2, cell_mmd = metrics(source_adata, target_adata, gene_list=gene_list, data_space='X')
            embedding_r2, embedding_l2, embedding_mmd = np.nan, np.nan, np.nan, np.nan

            return embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd
        
        elif self.model_name == 'observed':
            embedding_r2, embedding_l2, embedding_mmd = np.nan, np.nan, np.nan, np.nan
            cell_r2, cell_l2, cell_mmd = metrics(source_adata, target_adata, gene_list=gene_list, data_space='X')
            return embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd
        
        else:
            
            transported_adata = self.predict(source_adata)
            
            if self.embedding:
                target_adata.obsm['X_emb'] = self.embedding_model.get_latent_representation(target_adata)

            if target_adata.X.max() > 50:
                print("Log transforming target data", transported_adata.X.max(), target_adata.X.max())
                target_adata.X = normalize_and_log_transform(target_adata.X)

            if self.embedding:
                embedding_r2, embedding_l2, embedding_mmd = metrics(transported_adata, target_adata,  data_space='X_emb')
            else:
                embedding_r2, embedding_l2, embedding_mmd = np.nan, np.nan, np.nan, np.nan

            cell_r2, cell_l2, cell_mmd = metrics(transported_adata, target_adata, gene_list=gene_list, data_space='X')

            return embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd



import os
import pandas as pd
import numpy as np
import anndata as ad
import ray
from typing import List



@ray.remote(
    num_cpus=2,
    num_gpus=1,
    max_retries=-1,        # Infinite retries for worker failures
    memory=1.6e+10,
)
def run_perturbation_task(
    adata: AnnData,
    datasets_name: str,
    output_dir: str,
    perturbation_attribute: str,
    test_size: float,
    device: str,
    embedding: bool,
    latent_dims: List[int],
    num_run: int,
    start_run: int,
    perturbation: str
):
    import os
    import pandas as pd
    import numpy as np
    import anndata as ad

    print(f"Running task for {perturbation}")

    for run in range(1, num_run + 1):
        run_output_dir = os.path.join(
            output_dir, datasets_name, f"{perturbation}_{run + start_run}"
        )
        if not os.path.exists(run_output_dir):
            os.makedirs(run_output_dir)

        # Check if data split exists
        if all(os.path.exists(os.path.join(run_output_dir, f'{split}.h5ad'))
               for split in ['source_train', 'source_test', 'target_train', 'target_test']):
            print(f"Loading existing data split for {perturbation}, run {run}")
            source_train_adata = ad.read_h5ad(os.path.join(run_output_dir, 'source_train.h5ad'))
            source_test_adata = ad.read_h5ad(os.path.join(run_output_dir, 'source_test.h5ad'))
            target_train_adata = ad.read_h5ad(os.path.join(run_output_dir, 'target_train.h5ad'))
            target_test_adata = ad.read_h5ad(os.path.join(run_output_dir, 'target_test.h5ad'))
        else:
            print(f"Creating new data split for {perturbation}, run {run}")
            # Split the data into source (control) and target (perturbation)
            source_adata = adata[adata.obs[perturbation_attribute] == 'control'].copy()
            target_adata = adata[adata.obs[perturbation_attribute] == perturbation].copy()

            # Randomly shuffle the data
            source_adata = source_adata[np.random.permutation(len(source_adata))]
            target_adata = target_adata[np.random.permutation(len(target_adata))]

            # Ensure test sets have at least 500 cells and no more than 1000 cells
            source_test_size = max(500, min(1000, int(len(source_adata) * test_size)))
            target_test_size = max(500, min(1000, int(len(target_adata) * test_size)))

            source_train_adata = source_adata[:-source_test_size].copy()
            source_test_adata = source_adata[-source_test_size:].copy()
            target_train_adata = target_adata[:-target_test_size].copy()
            target_test_adata = target_adata[-target_test_size:].copy()

            # Save the split data
            source_train_adata.write(os.path.join(run_output_dir, 'source_train.h5ad'))
            source_test_adata.write(os.path.join(run_output_dir, 'source_test.h5ad'))
            target_train_adata.write(os.path.join(run_output_dir, 'target_train.h5ad'))
            target_test_adata.write(os.path.join(run_output_dir, 'target_test.h5ad'))

        # List to store results for this run
        run_results = []

        for latent_dim in latent_dims:
            for model_name in ['identity', 'observed', 'scgen', 'w1ot', 'w2ot']:

                model_output_dir = os.path.join(
                    run_output_dir,
                    f'saved_models',
                    model_name
                )
                if not os.path.exists(model_output_dir):
                    os.makedirs(model_output_dir)

                # Initialize and train the model
                if '4i' in datasets_name:
                    if model_name == 'scgen':
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=8,
                            embedding=True,
                            output_dir=model_output_dir,
                            hidden_layers=[32, 32],
                            num_iters=10000,
                            device=device
                        )

                        pmodel.train()

                        # Evaluate the model
                        metrics = pmodel.evaluate(
                            source_test_adata,
                            target_test_adata
                        )
                        embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd = metrics

                    elif model_name == 'identity':
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=latent_dim,
                            embedding=False,
                            output_dir=model_output_dir,
                            device=device
                        )

                        # Evaluate the model
                        metrics = pmodel.evaluate(
                            source_test_adata,
                            target_test_adata
                        )
                        embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd = metrics
                    
                    elif model_name == 'observed':
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=latent_dim,
                            embedding=False,
                            output_dir=model_output_dir,
                            device=device
                        )

                        # Evaluate the model
                        metrics = pmodel.evaluate(
                            target_train_adata,
                            target_test_adata
                        )
                        embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd = metrics
                    else:
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=latent_dim,
                            embedding=False,
                            output_dir=model_output_dir,
                            device=device
                        )

                        pmodel.train()

                        metrics = pmodel.evaluate(
                            source_test_adata,
                            target_test_adata
                        )
                        embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd = metrics
                elif datasets_name in ['sciplex3-hvg-top1k']:
                    if model_name == 'scgen':
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=50,
                            embedding=True,
                            output_dir=model_output_dir,
                            hidden_layers=[512, 512],
                            num_iters=250000,
                            device=device
                        )

                        pmodel.train()

                        # Evaluate the model
                        metrics = pmodel.evaluate(
                            source_test_adata,
                            target_test_adata
                        )
                        embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd = metrics

                    elif model_name == 'identity':
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=latent_dim,
                            embedding=False,
                            output_dir=model_output_dir,
                            device=device
                        )

                        # Evaluate the model
                        metrics = pmodel.evaluate(
                            source_test_adata,
                            target_test_adata
                        )
                        embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd = metrics
                    
                    elif model_name == 'observed':
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=latent_dim,
                            embedding=False,
                            output_dir=model_output_dir,
                            device=device
                        )

                        # Evaluate the model
                        metrics = pmodel.evaluate(
                            target_train_adata,
                            target_test_adata
                        )
                        embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd = metrics
                    else:
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=latent_dim,
                            embedding=False,
                            output_dir=model_output_dir,
                            device=device
                        )

                        pmodel.train()

                        metrics = pmodel.evaluate(
                            source_test_adata,
                            target_test_adata
                        )
                        embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd = metrics

                else:
                    if model_name == 'identity':
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=latent_dim,
                            embedding=False,
                            output_dir=model_output_dir,
                            device=device
                        )

                        # Evaluate the model
                        metrics = pmodel.evaluate(
                            source_test_adata,
                            target_test_adata
                        )
                        embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd = metrics
                    
                    elif model_name == 'observed':
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=latent_dim,
                            embedding=False,
                            output_dir=model_output_dir,
                            device=device
                        )

                        # Evaluate the model
                        metrics = pmodel.evaluate(
                            target_train_adata,
                            target_test_adata
                        )
                        embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd = metrics
                    else:
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=latent_dim,
                            embedding=embedding,
                            output_dir=model_output_dir,
                            device=device
                        )

                        pmodel.train()

                        metrics = pmodel.evaluate(
                            source_test_adata,
                            target_test_adata
                        )
                        embedding_r2, embedding_l2, embedding_mmd, cell_r2, cell_l2, cell_mmd = metrics

                

                # Save the results
                model_results = {
                    'model': model_name,
                    'perturbation': perturbation,
                    'latent_dim': latent_dim,
                    'cell_r2': cell_r2, 
                    'cell_l2': cell_l2,
                    'cell_mmd': cell_mmd,
                }
                run_results.append(model_results)

        # Save the run results to a CSV file
        results_df = pd.DataFrame(run_results)
        results_df.to_csv(os.path.join(run_output_dir, 'results.csv'), index=False)
    
    print(f"Finished running task for {perturbation}")


def run_iid_perturbation(
    dataset_path: str,
    output_dir: str = './iid_experiments/',
    perturbation_attribute: str = 'perturbation',
    test_size: float = 0.2,
    device: str = 'cuda',
    embedding: bool = True,
    latent_dims: List[int] = [50],
    num_run: int = 1,
    start_run: int = 0,
):
    import os
    import pandas as pd
    import numpy as np
    import anndata as ad

    datasets_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the dataset once and put it into Ray's object store
    adata = ad.read_h5ad(dataset_path)
    adata_ref = ray.put(adata)

    # Get unique perturbations, excluding 'control'
    perturbations = [
        p for p in adata.obs[perturbation_attribute].unique() if p != 'control'
    ]
    if 'sciplex3-hvg' in datasets_name:
        perturbations = ['belinostat', 'dacinostat', 'givinostat',
                        'hesperadin', 'tanespimycin', 'jnj_26854165',
                        'tak_901', 'flavopiridol_hcl', 'alvespimycin_hcl']
    # Submit tasks to Ray for each perturbation
    tasks = []
    for perturbation in perturbations:
        print(f"Submitting task for {perturbation}")
        task = run_perturbation_task.remote(
            adata_ref,
            datasets_name,
            output_dir,
            perturbation_attribute,
            test_size,
            device,
            embedding,
            latent_dims,
            num_run,
            start_run,
            perturbation
        )
        tasks.append(task)

    return tasks

    