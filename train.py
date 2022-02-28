from tqdm import tqdm

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from utils.dataloader import *

DATASET_PATH = 'dataset/k_youtube_data.csv'
MODEL_PATH = 'skt/kogpt2-base-v2'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv(DATASET_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)

# train dataset 만들기
docs = df['vid_name']

# padding할 max_length 구하기
max_length = 0

for doc in docs:
    text = '<s>' + doc + '</s>'
    encoded = tokenizer.encode(text)
    max_length = max(max_length, len(encoded))

# make dataset
dataset = TitleDataset(docs, tokenizer, max_length)

# train_model
def train_model(model, dataset, batch_size=32, epochs=4, learning_rate=3e-5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for data in tqdm(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            outputs = model(data, labels=data)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    
    model.save_pretrained('model_file/model.bin')
    
    return model

if __name__ == '__main__':
    train_model(model, dataset)