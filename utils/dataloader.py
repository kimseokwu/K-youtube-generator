import torch
from torch.utils.data import Dataset, DataLoader

class TitleDataset(Dataset):
    def __init__(self, docs, tokenizer, max_length):
        self.docs = docs
        self.tokenizer = tokenizer
        self.data = []
        
        for doc in docs:
            doc = self.tokenizer.bos_token + doc + self.tokenizer.eos_token
            encoding_dict = tokenizer(doc, max_length=max_length, padding='max_length')
            self.data.append(torch.tensor(encoding_dict['input_ids']))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]