from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.datasets import *


# train_model
def train_model(model, dataset, batch_size=32, epochs=4, learning_rate=3e-5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
    
        print(f'EPOCH: {epoch}, loss: {loss.item()}')
        model.save_pretrained(f'model_file/model_{epoch}')
    
    return model