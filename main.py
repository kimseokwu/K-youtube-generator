import pandas as pd
import torch
from generator import *
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

import train
import utils.datasets as datasets

MODEL_PATH = 'skt/kogpt2-base-v2'
FILE_PATH = 'dataset/k_youtube_datasets.csv'
EPOCH = 4
BATCH_SIZE = 32
lr = 0.00003

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH,
                                                    bos_token='<s>',
                                                    eos_token='</s>',
                                                    unk_token='<unk>',
                                                    pad_token='<pad>',
                                                    mask_token='<mask>')

data = pd.read_csv(FILE_PATH)
docs = data['vid_name']
train_dataset = datasets.TitleDataset(docs, tokenizer)

# train model
model = train.train_model(model, train_dataset, batch_size=BATCH_SIZE, epochs=EPOCH, learning_rate=lr)
model.save_pretrained('model_file/final_model')

# load model
checkpoint = torch.load('model_file/final_model/pytorch_model.bin', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

if __name__ == "__main__":
    text = input('input text: ')
    for i in range(10):
        sent = generate_title(model, tokenizer, text)
        print(sent)