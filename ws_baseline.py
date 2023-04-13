import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import torch
from utils import create_transcript, ridge_n_plot, plot_stat_map, create_words_activations_dict
import pickle
import seaborn as sns

def embedding_window_size(words,ws, HF_model="gpt2", device='cpu',printi=False, mean_last_hidden=False):
    import numpy as np
    from tqdm import tqdm
    
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    
    tokenizer = GPT2Tokenizer.from_pretrained(HF_model)
    model = GPT2Model.from_pretrained(HF_model).to(device)

    
    vectors=[]
    for i in tqdm(range(len(words))):
        j= i-ws if i>ws else 0
        text=" ".join(words[j:i+1])
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(inputs.input_ids)
        if mean_last_hidden:
            last_word_embedding = np.mean(outputs[0].to("cpu").detach().numpy()[0], axis=0)
        else:
            last_word_embedding = outputs[0].to("cpu").detach().numpy()[0,-1,:]
        vectors.append(last_word_embedding)

    return np.array(vectors)

def compare_ws(wss, words, y, non_zero_activity_indices, transcript, non_zero_activity_dict, device='cpu'):
    corrs_dict = {}

    for i, ws in enumerate(wss):
        print(f'ws = {ws}')
        embds = embedding_window_size(words, ws=ws, mean_last_hidden=True, device=device)
        embds_lst = [v for v in embds]
        transcript['embds'] = embds_lst
        TR_embds = transcript[['TR_onset', 'embds']].groupby('TR_onset').mean() # For each TR average the embeddings of all words that fall under this TR

        X_comapre_ws = np.squeeze(np.array(TR_embds.values.tolist()))

        wt, corrs_compare_ws = ridge_n_plot(X_comapre_ws, y, print_plot=False)
        corrs_dict[ws] = corrs_compare_ws

        corrs_total_brain = np.zeros(40962*2)
        for non_zero_idx in non_zero_activity_indices:
            corrs_total_brain[non_zero_idx] = corrs_compare_ws[non_zero_activity_dict[non_zero_idx]]
            # corrs_total_brain[non_zero_idx] = 0
        plot_stat_map(corrs_total_brain, threshold=0.25)

        
    df = pd.DataFrame(corrs_dict)
    df.index.name = "Window size"

    g = sns.displot(df, stat="probability", common_norm=False, kde=True, palette="crest")
    g.set_axis_labels("Correlation", "Proportion of voxels in this interval")
    g._legend.set_title("Window size")


def main():
    # create_words_activations_dict(save_to_pkl=True)
    with open('/home/meiri.yoav/w2b/embeddings_and_activations/words_activations_dict.pkl', 'rb') as f:
        words_activations_dict = pickle.load(f)
    
    words = words_activations_dict['words']
    y = words_activations_dict['y']
    non_zero_activity_indices = words_activations_dict['non_zero_activity_indices']
    non_zero_activity_dict = words_activations_dict['non_zero_activity_dict']
    
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
    transcript = create_transcript(task=task, sub=sub)['transcript']
    transcript['word'] = transcript['word'].apply(lambda x:  x if x != '<unk>' else tokenizer.unk_token)
    
    #----------------------------------------------------------------------------------------------------
    embds = embedding_window_size(words, ws=16, mean_last_hidden=True, device=device)
    embds_lst = [v for v in embds]
    transcript['embds'] = embds_lst
    TR_embds = transcript[['TR_onset', 'embds']].groupby('TR_onset').mean() # For each TR average the embeddings of all words that fall under this TR

    X_ws = np.squeeze(np.array(TR_embds.values.tolist()))
    wt, corrs_ws = ridge_n_plot(X_ws, y, print_plot=False)
    
    corrs_total_brain = np.zeros(40962*2)
    for non_zero_idx in non_zero_activity_indices:
        corrs_total_brain[non_zero_idx] = corrs_ws[non_zero_activity_dict[non_zero_idx]]
    plot_stat_map(corrs_total_brain, threshold=0.25)
    
    #----------------------------------------------------------------------------------------------------
    # # export results to pkl
    # with open('X_ws.pkl', 'wb') as f:
    #     pickle.dump(X_ws, f)
    
    
    #----------------------------------------------------------------------------------------------------
    # Compare different window sizes:
    compare_ws([0, 4, 8, 16, 32, 64, 128], words, y, non_zero_activity_indices,
               transcript, non_zero_activity_dict, device=device) # 0 means embed only the last word heard
    
if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()