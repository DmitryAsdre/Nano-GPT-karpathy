import pickle
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers.bpe_tokenizer import BPETokenizer
from tokenizers.char_tokenizer import CharTokenizer

def save_model(path, model_name, model, tokenizer):
    model_dir = os.path.join(path, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(path, model_name, 'model.pt')
    tokenizer_path = os.path.join(path, model_name, 'tokenizer.pickle')
    
    torch.save(model.state_dict(), model_path)
    
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
        
def load_model(path, model_name, model):
    model_dir = os.path.join(path, model_name)
    
    state_dict = torch.load(os.path.join(model_dir, 'model.pt'))
    with open(os.path.join(model_dir, 'tokenizer.pickle'), 'rb') as f:
        tokenizer = pickle.load(f)
        
    model.load_state_dict(state_dict)
    return model, tokenizer

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, 
                  train_data, 
                  val_data, 
                  eval_iters,
                  block_size,
                  batch_size, 
                  device):
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == 'train':
                data = train_data
            else:
                data = val_data
            X, Y = get_batch(data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    
    return out

def train_model(model,
                tokenizer,
                optimizer,
                max_iters,
                eval_interval,
                train_data, 
                val_data, 
                eval_iters,
                block_size,
                batch_size, 
                model_path,
                model_name,
                device):
    train_losses = []
    val_losses = []
    
    model = model.to(device)
    cur_best_loss = float('inf')
    
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, eval_iters, block_size, batch_size, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            train_losses.append(losses['train'].item())
            val_losses.append(losses['val'].item())
            
            if cur_best_loss > losses['val']:
                cur_best_loss = losses['val']
                save_model(model_path, model_name, model, tokenizer)
                print('Saved model!')
                
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        
        loss.backward()
        optimizer.step()
        
    model, tokenizer = load_model(model_path, model_name, model)
    
    return model, tokenizer,\
           train_losses, val_losses

@torch.no_grad()
def generate(input_text,
             model,
             tokenizer,
             max_new_tokens,
             device):
    context = tokenizer.encode(input_text)
    context = torch.tensor(context, dtype=torch.long, device=device).view(1, -1)
    gen = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    return gen, tokenizer.decode(gen)