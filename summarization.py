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
from transformers import pipeline

def embedding_ws_summary(words,ws, HF_model="gpt2", device='cpu',printi=False, summerize_from=20, update_every=30):
    tokenizer = GPT2Tokenizer.from_pretrained(HF_model)
    model = GPT2Model.from_pretrained(HF_model).to(device)

    hf_name = 'pszemraj/led-large-book-summary'
    summarizer = pipeline(
    "summarization",
    hf_name,
    device=device,
    )

    vectors=[]
    for i in tqdm(range(len(words))):
        j= i-ws if i>ws else 0
        text=" ".join(words[j:i+1])
        summery = "This is a summary of the text so far: "
        if i+1 > summerize_from:
            if i % update_every == 0 or i == summerize_from:
                l_idx = max([i+1 - 500, 0]) if i+1 > 50 else 0
                text_summary = " ".join(words[l_idx:i+1])
                summery = "This is a summary of the text so far: " + summarizer(summery +" The continuation is: "+ text_summary)[0]['summary_text']
            input_t_w_summary = tokenizer(summery + ' : ' + text, return_tensors="pt").to(device)
            outputs_t_w_summary = model(input_t_w_summary.input_ids)
            summary_t_w_embedding = outputs_t_w_summary[0].to("cpu").detach().numpy()[0,-1,:]
            
            vectors.append(summary_t_w_embedding)
        else:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(inputs.input_ids)
            last_word_embedding = outputs[0].to("cpu").detach().numpy()[0,-1,:]
            vectors.append(last_word_embedding)

    return np.array(vectors)


def main():
    # create_words_activations_dict(save_to_pkl=True)
    with open('words_activations_dict.pkl', 'rb') as f:
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
    embds = embedding_ws_summary(words,16, HF_model="gpt2", device=device)
    embds_lst = [v for v in embds]
    transcript['embds'] = embds_lst
    TR_embds = transcript[['TR_onset', 'embds']].groupby('TR_onset').mean() # For each TR average the embeddings of all words that fall under this TR

    X_w_summary = np.squeeze(np.array(TR_embds.values.tolist()))
    
    # import pickle
    # with open('/home/meiri.yoav/X_w_summary.pkl', 'rb') as f:
    #     X_w_summary = pickle.load(f)
    #----------------------------------------------------------------------------------------------------
    # plot correlation histogram and summary statistics
    wt, corrs_summery = ridge_n_plot(np.vstack((X_w_summary)), y)
    print(f'{round(len([x for x in corrs_summery if x > 0.3]) / len(corrs_summery) * 100, 2)} % are bigger than 0.3')
    
    # plot stat map -> summarization
    corrs_total_brain = np.zeros(2*40962)
    for non_zero_idx in non_zero_activity_indices:
        corrs_total_brain[non_zero_idx] = corrs_summery[non_zero_activity_dict[non_zero_idx]]
    plot_stat_map(corrs_total_brain, threshold=0.25)
    
    # plot stat map - pred_cont sub by baseline
    with open('X_ws.pkl', 'rb') as f:
        X_ws = pickle.load(f)
    corrs_ws = ridge_n_plot(X_ws, y)[1]
    
    corrs_total_brain = np.zeros(2*40962)
    corrs_diff = corrs_summery - corrs_ws
    for non_zero_idx in non_zero_activity_indices:
        corrs_total_brain[non_zero_idx] = corrs_diff[non_zero_activity_dict[non_zero_idx]]
    plot_stat_map(corrs_total_brain, threshold=0.15)

if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()