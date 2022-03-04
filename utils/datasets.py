import torch
from torch.utils.data import Dataset

class TitleDataset(Dataset):
    def __init__(self, docs, tokenizer, vector_length=67):
        self.docs = docs
        self.tokenizer = tokenizer
        self.data = []
        
        for doc in docs:
            doc = self.tokenizer.bos_token + doc + self.tokenizer.eos_token
            encoding_dict = tokenizer(doc, max_length=vector_length, padding='max_length')
            self.data.append(torch.tensor(encoding_dict['input_ids']))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]