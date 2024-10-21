if __name__ == '__main__':
    
    
    dataset_path = '/fs/cbcb-scratch/cys/w1ot/datasets/4i-melanoma-8h-48.h5ad'

    import ray
    from w1ot.experiments import run_iid_perturbation
    
    ray.init(address='auto')
    
    tasks = run_iid_perturbation(
        dataset_path=dataset_path,
        output_dir='/fs/cbcb-scratch/cys/w1ot/iid_experiments/',
        perturbation_attribute='drug',
        test_size=0.2,
        device='cuda',
        embedding=True,
        latent_dims=[8],
        num_run=5,
        start_run=0,
    )

    ray.get(tasks)
    
    ray.shutdown()