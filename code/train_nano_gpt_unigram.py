import os
import pickle

import torch
import matplotlib.pyplot as plt

from nano_gpt.model import NanoGPT
from nano_gpt.utils import train_model
from nano_gpt.utils import generate

from tokenizers.unigram_tokenizer import UnigramTokenizer
class CFG:
    batch_size = 32
    block_size = 384
    max_iters = 15_002
    eval_interval = 1000
    learning_rate = 3e-4
    device = 'cuda:0'
    eval_iters = 400
    n_embed = 384
    num_heads = 12
    num_layers = 12
    ffwd_coef = 4
    dropout = 0.25
    random_state = 42
    vocab_size = 1_000
    initial_vocab_multiplier = 12
    
    model_path = '../models/'
    model_name = 'nano_gpt_unigram_1000'
    input_data = '../data/extended_input.txt'
    
    test_size = 0.1
    
torch.manual_seed(CFG.random_state)

if __name__ == '__main__':
    ## Load data
    with open(CFG.input_data, 'r') as f:
        data = f.read()
        
    ## Train tokenizer
    tokenizer = UnigramTokenizer(CFG.vocab_size, CFG.initial_vocab_multiplier)
    tokenizer.train_tokenizer([data])
    tokens = tokenizer.encode(data)

    ## Train, test split
    data = torch.tensor(tokens, dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    ## Create NanoGPT model and optimizer
    model = NanoGPT(tokenizer.vocab_size,
                    CFG.block_size,
                    CFG.n_embed,
                    CFG.num_heads,
                    CFG.dropout,
                    CFG.ffwd_coef,
                    CFG.num_layers).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate)
    
    
    ## Train model
    model, tokenizer, train_losses, val_losses = train_model(model, tokenizer, optimizer,
                                                        CFG.max_iters, CFG.eval_interval, train_data,
                                                        val_data, CFG.eval_iters, CFG.block_size,
                                                        CFG.batch_size, CFG.model_path, CFG.model_name, CFG.device)
    
    ## Save losses
    with open(os.path.join(CFG.model_path, CFG.model_name, 'train_losses.pickle'), 'wb') as f:
        pickle.dump(train_losses, f)
        
    with open(os.path.join(CFG.model_path, CFG.model_name, 'val_losses.pickle'), 'wb') as f:
        pickle.dump(val_losses, f)
    
    ## Examples
    initial_context = "OTHELLO.\nHave you pray'd to-night, Desdemona?\n"
    res = generate(initial_context, model, tokenizer, 350, CFG.device)

    print(res)
    
    initial_context = "CHAMBERLAIN.\n"
    res = generate(initial_context, model, tokenizer, 350, CFG.device)

    print(res)
    
    initial_context = "OTHELLO.\n"
    res = generate(initial_context, model, tokenizer, 350, CFG.device)

    print(res)