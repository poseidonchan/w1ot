import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_and_merge_results(root_dir='iid_experiments'):
    all_results = []
    for dataset_name in os.listdir(root_dir):
        dataset_dir = os.path.join(root_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for perturbation_run in os.listdir(dataset_dir):
            perturbation_dir = os.path.join(dataset_dir, perturbation_run)
            if not os.path.isdir(perturbation_dir):
                continue
            results_file = os.path.join(perturbation_dir, 'results.csv')
            if os.path.isfile(results_file):
                df = pd.read_csv(results_file)
                df['dataset_name'] = dataset_name
                df['perturbation'] = perturbation_run.split('_')[0]
                all_results.append(df)
    merged_df = pd.concat(all_results, ignore_index=True)
    return merged_df

def plot_metrics_by_dataset(df, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    metrics = ['cell_r2', 'cell_l2', 'cell_mmd', 'cell_wasserstein']
    
    colormap = {
        'w1ot': '#51A9D6',
        'w2ot': '#E05732',
        'scgen': '#7C7379',
        'identity': '#C5BEBE',
        'observed': '#9FA6C3'
    }
    
    perturbations = ['belinostat', 'dacinostat', 'givinostat',
                     'hesperadin', 'tanespimycin', 'jnj_26854165',
                     'tak_901', 'flavopiridol_hcl', 'alvespimycin_hcl']
    
    datasets = df['dataset_name'].unique()
    for dataset in datasets:
        df_dataset = df[df['dataset_name'] == dataset]
        if 'sciplex3' in dataset:
            df_dataset = df_dataset[df_dataset['perturbation'].isin(perturbations)]
        dataset_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='model', y=metric, data=df_dataset, palette=colormap)
            plt.title(f'{dataset}: Plot of {metric}')
            plt.xlabel('Model')
            plt.ylabel(metric)

            if 'mmd' in metric:
                plt.yscale('log')
                plt.ylim(bottom=0)
            
            elif 'r2' in metric:
                plt.ylim(bottom=0, top=1)
            
            elif 'l2' in metric:
                plt.ylim(bottom=0)
            
            plt.tight_layout()
            plt.savefig(os.path.join(dataset_dir, f'{metric}_plot.png'))
            plt.close()

if __name__ == '__main__':
    merged_df = read_and_merge_results()
    plot_metrics_by_dataset(merged_df)