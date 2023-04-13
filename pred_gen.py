import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import torch
from utils import create_transcript, ridge_n_plot, plot_stat_map
import pickle
import logging
logging.disable(logging.WARNING)
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
from tqdm import tqdm
from transformers import pipeline, set_seed

def embedding_ws_pred_cont(words,ws, HF_model="gpt2", device='cpu',printi=False, mean_last_hidden=True, update_every=10):

    set_seed(42)

    print_for = 5
    tokenizer = GPT2Tokenizer.from_pretrained(HF_model, padding_side="left")
    model = GPT2Model.from_pretrained(HF_model).to(device)

    generator = pipeline('text-generation', model='gpt2')
    vectors=[]
    for i in tqdm(range(len(words))):
        j= i-ws if i>ws else 0
        text=" ".join(words[j:i+1])
        
        if i % update_every == 0:
            l_idx = i+1 - min([30, i+1]) if i+1 > 16 else 0
            pred_cont = generator(" ".join(words[l_idx : i+1]), max_length=50)[0]['generated_text']

        text = pred_cont + '. This is the continuation of: ' + text
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(inputs.input_ids)
        if mean_last_hidden:
            last_word_embedding = np.mean(outputs.last_hidden_state.to("cpu").detach().numpy()[0], axis=0)
        else:
            last_word_embedding = outputs.last_hidden_state.to("cpu").detach().numpy()[0,-1,:]
    
        vectors.append(last_word_embedding)

    return vectors


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
    # Get the embeddings of the predicted continuation
    pred_embds = embedding_ws_pred_cont(words,16, HF_model="gpt2", device=device, mean_last_hidden=True, update_every=1)
    embds_lst = [v for v in pred_embds]
    transcript['embds'] = embds_lst
    TR_embds = transcript[['TR_onset', 'embds']].groupby('TR_onset').mean() # For each TR average the embeddings of all words that fall under this TR

    X_pred_cont = np.squeeze(np.array(TR_embds.values.tolist()))
    
    #----------------------------------------------------------------------------------------------------
    # plot correlation histogram and summary statistics
    wt, corrs_pred = ridge_n_plot(X_pred_cont, y)
    pd.Series(corrs_pred).describe()
    
    # plot stat map - pred_cont
    corrs_total_brain = np.zeros(40962*2)
    for non_zero_idx in non_zero_activity_indices:
        corrs_total_brain[non_zero_idx] = corrs_pred[non_zero_activity_dict[non_zero_idx]]
    plot_stat_map(corrs_total_brain, threshold=0.25)
    
    # plot stat map - pred_cont sub by baseline
    with open('/home/meiri.yoav/w2b/embeddings_and_activations/X_ws.pkl', 'rb') as f:
        X_ws = pickle.load(f)
    corrs_ws = ridge_n_plot(X_ws, y)[1]
    
    corrs_total_brain = np.zeros(40962*2)
    corrs_diff = corrs_pred - corrs_ws
    for non_zero_idx in non_zero_activity_indices:
        corrs_total_brain[non_zero_idx] = corrs_diff[non_zero_activity_dict[non_zero_idx]]
    plot_stat_map(corrs_total_brain, threshold=0.2)

if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()