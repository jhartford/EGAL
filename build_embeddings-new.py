import transformers
from tqdm import tqdm

import glob
import logging

import numpy as np
import sys
import torch
import pickle

from transformers import (
    BertModel,
    BertTokenizer
)

def text_to_embedding_fn(text, tokenizer, model):
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
    outputs = model(input_ids)
    return outputs

def main():
    words = [i.split('/')[-1].split('.')[0] for i in glob.glob('./data/parsed/*.csv')]
    if len(sys.argv) > 1:
        words = [words[int(sys.argv[1])]]

    model_type = 'bert'
    model_weights = 'bert-large-cased'

    tokenizer = BertTokenizer.from_pretrained(model_weights)
    model = BertModel.from_pretrained(model_weights)
    text_to_embedding = lambda text: text_to_embedding_fn(text, tokenizer, model)

    id_lookup = {w:tokenizer.encode(w, add_special_tokens=False) for w in words}

    for w in tqdm(words):
        # skip words with multiple ids
        if len(id_lookup[w]) > 1:
            continue
        target_csv = open(f"./data/parsed/{w}.csv").readlines()
        embeddings = []
        target = []
        pbar = tqdm(target_csv)
        pbar.set_description(w)
        for i, line in enumerate(pbar):
            with torch.no_grad():
                try:
                    text = line.strip().split('<<<>>>')[1].strip()
                except IndexError:
                    #print(line)
                    continue
                try:
                    target_location = tokenizer.encode(text.lower(), add_special_tokens=True).index(id_lookup[w][0])
                    outputs = text_to_embedding(text)
                    embeddings.append(outputs[0][:,target_location,:].flatten().detach().cpu().numpy())
                    target.append(line.strip().split('<<<>>>')[0].strip())
                except:
                    #tqdm.write(f"Skipping {i}: {text}")
                    #raise
                    continue
        classes = np.unique(np.array(target))
        target_id = np.zeros_like(target, dtype='int')
        for i,c in enumerate(classes):
            target_id[np.array(target) == c] = i
        np.savez_compressed(f'./embeddings/{w}.npz', embeddings = np.array(embeddings), target=target_id, classes=classes)

if __name__ == '__main__':
    main()
