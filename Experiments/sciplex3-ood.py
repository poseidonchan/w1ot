if __name__ == '__main__':
    dataset_path = '/fs/cbcb-scratch/cys/w1ot/datasets/sciplex3-hvg-top1k_50.h5ad'

    import ray
    from w1ot.experiments import run_ood_perturbation
    
    # Initialize Ray with default temp directory
    ray.init(address='auto')
    
    # Run OOD experiments using cell_type as the OOD attribute
    tasks = run_ood_perturbation(
        dataset_path=dataset_path,
        ood_attribute='cell_type',  # Use cell_type as the OOD attribute
        output_dir='/fs/cbcb-scratch/cys/w1ot/ood_experiments/',
        perturbation_attribute='drug',
        device='cuda',
        embedding=True,
        latent_dims=[50],
        num_run=5,  # Run 5 experiments with different random splits
        start_run=0,
    )

    # Wait for all tasks to complete
    ray.get(tasks)
    
    # Shutdown Ray
    ray.shutdown() 