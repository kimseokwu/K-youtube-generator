import argparse

import pandas as pd
import torch
from generator import *
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

import train
import utils.datasets as datasets

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=4, help='n of epochs for fine tuning')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for datasets')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate for fine tuning')
parser.add_argument('--data_file_path', type=str, default='dataset/k_youtube_datasets.csv', help='path where load datasets')
parser.add_argument('--save_path', type=str, default='model_file/final_model', help='path where save the tuned model file')
parser.add_argument('--load_path', type=str, default='model_file/final_model/pytorch_model.bin', help='path where load the model file')
args = parser.parse_args()

MODEL_PATH = 'skt/kogpt2-base-v2'

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH,
                                                    bos_token='<s>',
                                                    eos_token='</s>',
                                                    unk_token='<unk>',
                                                    pad_token='<pad>',
                                                    mask_token='<mask>')

data = pd.read_csv(args.data_file_path)
docs = data['vid_name']
train_dataset = datasets.TitleDataset(docs, tokenizer)

# train model
model = train.train_model(model, train_dataset, batch_size=args.batch_size, epochs=args.epoch, learning_rate=args.lr)
model.save_pretrained(args.save_path)

# load model
checkpoint = torch.load(args.load_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

if __name__ == "__main__":
    text = input('input text: ')
    for i in range(10):
        sent = generate_title(model, tokenizer, text)
        print(sent)