from nilearn import surface
from nilearn import datasets as ndatasets
import numpy as np
from sklearn.model_selection import train_test_split
from ridge import bootstrap_ridge
import seaborn as sns
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2Model
import pickle as pkl

def create_words_activations_dict(save_to_pkl=False):
    HF_model="gpt2"
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(HF_model)
    participants=pd.read_table("/home/meiri.yoav/w2b_data/participants.tsv")
    task="tunnel"
    participants_in_task=[participants.participant_id[i] for i in range(len(participants)) if task in participants.task[i]]
    subs=[a.split("-")[1] for a in participants_in_task]

    # In the paper they suggested to exclude these participants
    subs.remove('004')
    subs.remove('013')

    sub=subs[2]
    first_TR=pd.read_table("/home/meiri.yoav/w2b_data/first_TRs/sub-"+sub+"_task-"+task+"_events.tsv").onset[0]


    transcript = create_transcript(task=task, sub=sub)['transcript']
    transcript['word'] = transcript['word'].apply(lambda x:  x if x != '<unk>' else tokenizer.unk_token)
    TRs = pd.unique(transcript['TR_onset'])
    
    # Get neural data
    all_neural_data = []
    for sub in subs:
        y_sub = get_neural_data(sub, task, TRs)
        all_neural_data.append(y_sub)
        
    all_neural_data = np.array(all_neural_data)
    y = np.nanmean(all_neural_data, axis=0)
    non_zero_activity_indices = [i for i in range(y.shape[1]) if np.any(y[:, i])]
    non_zero_activity_dict = {val: i for i, val in enumerate(non_zero_activity_indices)}
    y = y[:, non_zero_activity_indices] #*  leave only voxels with meassured activity

    zs = lambda v: (v-v.mean(0))/v.std(0) # normalize each voxel
    y = np.array([zs(tr) for tr in y])
    
    words = transcript['word'].to_list() # A list of all words in the story Tunnel
    
    words_activations_dict = {'words': words, 'y': y, 'non_zero_activity_indices': non_zero_activity_indices, 'non_zero_activity_dict': non_zero_activity_dict}

    if save_to_pkl:
        with open('words_activations_dict.pkl', 'wb') as f:
            pkl.dump(words_activations_dict, f)
            
    return words_activations_dict

def get_data(sub,task):
    """From a subject and a task creates a dictionary containing the subject's neural activity in both hemispheres that was collected during the task

    Args:
        sub (str): subject code. For example: '001'
        task (str): task name. For example: 'tunnel'

    Returns:
        Dict
    """
    # LOAD THE NEURAL DATA USING NILEARN
    neural_data_L=surface.load_surf_data("/home/meiri.yoav/w2b_data/afni-nonsmooth/sub-"+sub+"/func/sub-"+sub+"_task-"+task+"_space-fsaverage6_hemi-L_desc-clean.func.gii")
    neural_data_R=surface.load_surf_data("/home/meiri.yoav/w2b_data/afni-nonsmooth/sub-"+sub+"/func/sub-"+sub+"_task-"+task+"_space-fsaverage6_hemi-R_desc-clean.func.gii")
    
    surf_tpl=ndatasets.fetch_surf_fsaverage(mesh="fsaverage6")
    
    output={"neural_data_L":neural_data_L,"neural_data_R":neural_data_R,"fsaverage6_tpl":surf_tpl}
    return output

def create_transcript(task: str, sub: str):
    """Returns a dict which contains the task, the subject id, and a pandas dataframe where each records holds a word from the story, it's version after tokenization, it's onset and offset 
    in terms of TR's that passed from the moment the experiment began.

    Args:
        task (str): nerative name
        sub (str): subject id

    Returns:
        dict
    """
    import pandas as pd
    import numpy as np

    transcript=pd.read_csv(f"/home/meiri.yoav/w2b_data/{task}/align.csv",names=["word_orig","word","onset","offset"])
    transcript=transcript.dropna().reset_index(drop=True)
    onset = pd.read_table("/home/meiri.yoav/w2b_data/first_TRs/sub-"+sub+"_task-"+task+"_events.tsv").onset[0]
    TR=1.5
    transcript["TR_onset"]=(onset+transcript.onset)//TR
    transcript["TR_offset"]=(onset+transcript.offset)//TR
    transcript = transcript.astype({'TR_onset': 'int32', 'TR_offset': 'int32'})
    return {'transcript': transcript, 'task': task, 'sub': sub}


def get_neural_data(sub, task, TRs):
    import numpy as np
    import pandas as pd
    
    neural_data=get_data(sub,task)
    nd_concat = np.vstack((neural_data['neural_data_L'], neural_data['neural_data_R'])) # Voxal_num X No._Voxels
    # print(neural_data['neural_data_L'].shape, neural_data['neural_data_R'].shape, nd_concat.shape)
    return nd_concat.T[TRs]


def ridge_n_plot(X, y, print_plot=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    wt, corrs, valphas, allRcorrs, valinds = bootstrap_ridge(Rstim=X_train, Rresp=y_train, Pstim=X_test, Presp=y_test, alphas=np.logspace(0, 3, 20)
                                , nboots=5,chunklen=10,nchunks=int((0.2*844*0.8)/10),return_wt=True)

    if print_plot:
        g = sns.displot(corrs, stat="probability", common_norm=False, kde=True)
        g.set_axis_labels("Correlation", "Proportion of voxels in this interval")
    
    return wt, corrs

def plot_stat_map(corrs_total_brain, threshold=0.1, vmax=0.6, cmap='cold_hot'):
    from nilearn import plotting
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2,2,subplot_kw={"projection": "3d"},figsize=[15,10])
    plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

    
    SM=corrs_total_brain
    SM_l = SM[:40962]
    SM_r = SM[40962:]
    surf_tpl=ndatasets.fetch_surf_fsaverage(mesh="fsaverage6")
    mask_L=surface.load_surf_data("/home/meiri.yoav/w2b_data/afni-nonsmooth/tpl-fsaverage6/tpl-fsaverage6_hemi-L_desc-cortex_mask.gii")
    mask_R=surface.load_surf_data("/home/meiri.yoav/w2b_data/afni-nonsmooth/tpl-fsaverage6/tpl-fsaverage6_hemi-R_desc-cortex_mask.gii")

    plotting.plot_surf_stat_map(surf_tpl["pial_left"],SM_l*mask_L,hemi="left",view="lateral",colorbar=True,threshold=threshold,vmax=vmax,bg_map=surf_tpl["sulc_left"],axes=axes[0,0], cmap=cmap)
    plotting.plot_surf_stat_map(surf_tpl["pial_right"],SM_r*mask_R,hemi="right",view="lateral",colorbar=True,threshold=threshold,vmax=vmax,bg_map=surf_tpl["sulc_right"],axes=axes[0,1], cmap=cmap)
    plotting.plot_surf_stat_map(surf_tpl["pial_left"],SM_l*mask_L,hemi="left",view="medial",colorbar=True,threshold=threshold,vmax=vmax,bg_map=surf_tpl["sulc_left"],axes=axes[1,0], cmap=cmap)
    plotting.plot_surf_stat_map(surf_tpl["pial_right"],SM_r*mask_R,hemi="right",view="medial",colorbar=True,threshold=threshold,vmax=vmax,bg_map=surf_tpl["sulc_right"],axes=axes[1,1], cmap=cmap)
    plt.show()