import torch
from .generator import *
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

MODEL_PATH = 'skt/kogpt2-base-v2'

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH,
                                                    bos_token='<s>',
                                                    eos_token='</s>',
                                                    unk_token='<unk>',
                                                    pad_token='<pad>',
                                                    mask_token='<mask>')
checkpoint = torch.load('model_file/pytorch_model.bin')
model.load_state_dict(checkpoint)

if __name__ == "__main__":
    text = input('input text: ')
    sent = generate_title(model, tokenizer, text)
    print(sent)