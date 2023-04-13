from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from utils import ridge_n_plot, create_transcript, get_neural_data, plot_stat_map
import pickle
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import torch

def access_dim_reduction(X, max_dim, step):
    explained_var_vec = []
    
    for i in range(1, max_dim, step):
        pca = PCA(n_components=i)
        pca.fit(X)
        explained_var_vec.append(pca.explained_variance_ratio_.sum())
    
    df = pd.DataFrame({'Dimensions': list(range(1, max_dim, step)), 'Explained Variance': explained_var_vec})
    sns.lineplot(data=df, x='Dimensions', y='Explained Variance')

def dim_num_analysis(n_components_lst, X_ws, y):
    stat_dict = {}
    for n_components in n_components_lst:
        pca = PCA(n_components=n_components)
        pca.fit(X_ws)
        X_ws_reduced = X_ws @ pca.components_.T

        wt, corrs = ridge_n_plot(X_ws_reduced, y, print_plot=False)
        curr_stat_dict =  pd.DataFrame(pd.Series(corrs).describe()).to_dict()
        curr_stat_dict[0]['No. > 0.3'] = len([x for x in corrs if x > 0.3])
        stat_dict[n_components] = curr_stat_dict[0]
    return pd.DataFrame(stat_dict).T




def main():
    
    with open('X_ws.pkl', 'rb') as f:
        X_ws = pickle.load(f)
        
    
    with open('/home/meiri.yoav/w2b/embeddings_and_activations/words_activations_dict.pkl', 'rb') as f:
        words_activations_dict = pickle.load(f)
    
    y = words_activations_dict['y']
    non_zero_activity_indices = words_activations_dict['non_zero_activity_indices']
    non_zero_activity_dict = words_activations_dict['non_zero_activity_dict']
    access_dim_reduction(X_ws, 300, 1)
    n_components_lst = [16, 32, 64, 128, 256, 512, 768]
    dim_num_analysis(n_components_lst, X_ws, y)
    
    # Compare the stat maps for different dimensions (32 & 768)
    pca = PCA(n_components=32)
    pca.fit(X_ws)
    X_ws_reduced = X_ws @ pca.components_.T

    wt, corrs_32_pca = ridge_n_plot(X_ws_reduced, y, print_plot=False)
    print('Stat map dim=32')
    corrs_total_brain = np.zeros(2*40962)
    for non_zero_idx in non_zero_activity_indices:
        corrs_total_brain[non_zero_idx] = corrs_32_pca[non_zero_activity_dict[non_zero_idx]]
        # corrs_total_brain[non_zero_idx] = 0
    plot_stat_map(corrs_total_brain, threshold=0.25)

    wt, corrs_ws = ridge_n_plot(X_ws, y, print_plot=False)
    print('Stat map dim=768')
    corrs_total_brain = np.zeros(2*40962)
    for non_zero_idx in non_zero_activity_indices:
        corrs_total_brain[non_zero_idx] = corrs_ws[non_zero_activity_dict[non_zero_idx]]
        # corrs_total_brain[non_zero_idx] = 0
    plot_stat_map(corrs_total_brain, threshold=0.25)


if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()