import torch
from torch.functional import F 


def generate_title(model, tokenizer, text: str, max_length=30, temperature=0.7) -> str:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model.eval()
    count = 0
    sent = text[:]
    text = tokenizer.bos_token + text
    generated_token = ''
    
    with torch.no_grad():
        model = model.to(device)
        while generated_token != '</s>':
            
            if count > max_length:
                break
            
            input_ids = tokenizer.encode(text, return_tensors='pt')
            input_ids = input_ids.to(device)
            
            predicted = model(input_ids)
            pred = predicted[0]
            
            # temperature 적용
            logit = pred[:, -1, :] / temperature
            logit = F.softmax(logit, dim=-1)
            prev = torch.multinomial(logit, num_samples=1)
            generated_token = tokenizer.convert_ids_to_tokens(prev.squeeze().tolist())
            
            sent += generated_token.replace('▁', ' ')
            text += generated_token.replace('▁', ' ')
            count += 1
        
        sent = sent.replace('</s>', '')
        
        return sent