import re
from collections import defaultdict

import numpy as np
from .bpe_tokenizer import BPETokenizer



class UnigramTokenizer():
    def __init__(self, vocab_size, 
                 initial_vocab_multiplier,
                 shrink_multiplier=0.1,
                 sub_em_steps=2):
        self.vocab_size = vocab_size
        self.initial_vocab_multiplier = initial_vocab_multiplier
        self.shrink_multplier = shrink_multiplier
        self.sub_em_steps = sub_em_steps
        
        self.initial_tokenizer = BPETokenizer(int(self.vocab_size*self.initial_vocab_multiplier))
        
        self.cur_vocab_subword_freqs = None
        self.cur_vocab_subword_logprob = None
        
        self.alphabet = None
        
        self.stoi = {}
        self.itos = {}
    
    def pre_tokinze_str(self, text):
        return self.initial_tokenizer.pre_tokenize_text(text)
    
    def get_initial_word_freq(self):
        return self.initial_tokenizer.word_freq
    
    def train_initial_tokenizer(self, corpus):
        self.initial_tokenizer.train_tokenizer(corpus)
        self.alphabet = set(self.initial_tokenizer.smallest_vocab)
        
        self.cur_vocab_subword_freqs = self.initial_tokenizer.vocab_freqs
        tot_cnt = sum(list(self.cur_vocab_subword_freqs.values()))
        self.cur_vocab_subword_logprob = {k : np.log(v / tot_cnt) for k, v in self.cur_vocab_subword_freqs.items()}

    def get_initial_subword_logprob(self):
        vocab_freqs = self.initial_tokenizer.vocab_freqs
        tot_cnt = sum(list(vocab_freqs.values()))
        subword_logp = {k : np.log(v / tot_cnt) for k, v in vocab_freqs.items()}
        return subword_logp
          
    @staticmethod
    def viterbi_forward(word, subword_logp):
        best_subw_slices = [None]*(len(word) + 1)
        neg_loglik = np.zeros(len(word) + 1)
        
        for eow in range(1, len(word) + 1):
            neg_loglik[eow] = np.inf
            
            for bow in range(eow):
                subw = word[bow:eow]
                
                if subw in subword_logp:
                    logp = subword_logp[subw]
                    
                    s = neg_loglik[bow] - logp
                    if s < neg_loglik[eow]:
                        neg_loglik[eow] = s
                        best_subw_slices[eow] = (bow, eow)
        return neg_loglik, best_subw_slices
    
    @staticmethod
    def viterbi_backward(word, subw_slices, neg_loglik):
        subwords = []
        subwords_slices = []
        
        next_slices = subw_slices[-1]
        
        while next_slices is not None:
            subw = word[next_slices[0]:next_slices[1]]
            subwords.append(subw)
            subwords_slices.append((next_slices[0],next_slices[1]))
            next_slices = subw_slices[next_slices[0]]
        subwords.reverse()
    
        return subwords, subwords_slices, neg_loglik[-1]
    
    @staticmethod
    def get_viterbi_path(word, subword_logp):
        neg_loglik, best_subw_slices = UnigramTokenizer.viterbi_forward(word, subword_logp)
        subwords, subwords_slices, vit_path_loss = UnigramTokenizer.viterbi_backward(word, best_subw_slices, neg_loglik)
        
        return subwords, subwords_slices, vit_path_loss
    
    
    def run_e_step(self, estimated_logprob):
        initial_word_freq = self.get_initial_word_freq()
        
        viterbi_subword_freq = defaultdict(int)
        vit_path_loss_full = 0
        
        for word in initial_word_freq:
            word_freq = initial_word_freq[word]
            
            subwords_v, _, vit_path_loss = UnigramTokenizer.get_viterbi_path(word, estimated_logprob)
            vit_path_loss_full += vit_path_loss*word_freq
            for subword_v in subwords_v:
                viterbi_subword_freq[subword_v] += word_freq
        
        return  viterbi_subword_freq, vit_path_loss_full
    
    def run_m_step(self, viterbi_subword_freq):
        
        tot_cnt = sum(list(viterbi_subword_freq.values()))
        viterbi_logprob = {k : np.log(v / tot_cnt) for k, v in viterbi_subword_freq.items()}
        
        return viterbi_logprob
    
    
    def delta_loss(self, token, estimated_word_freqs, estimated_logprob):
        if token not in estimated_word_freqs:
            return None, np.inf
        
        if token in self.alphabet:
            return None, -np.inf
        
        if len(token) == 1:
            return None, -np.inf 
        
        most_probable_split = None
        most_probable_split_score = None
        
        token_logprob = estimated_logprob[token]
        estimated_logprob[token] = -np.inf
        
        most_probable_split, _, most_probable_split_score = UnigramTokenizer.get_viterbi_path(token, estimated_logprob)
        most_probable_split_score *= -1
        
        estimated_logprob[token] = token_logprob
                    
        if most_probable_split_score is None:
            return None, -np.inf
        
        return most_probable_split, \
               most_probable_split_score*estimated_word_freqs[token] - estimated_logprob[token]*estimated_word_freqs[token]
               
    def rebuid_vocab(self, tokens):
        new_subword_freqs = {}
        
        for token in tokens:
            new_subword_freqs[token] = self.cur_vocab_subword_freqs[token]
        self.cur_vocab_subword_freqs = new_subword_freqs
            
        tot_cnt = sum(list(self.cur_vocab_subword_freqs.values()))
        self.cur_vocab_subword_logprob = {k : np.log(v / tot_cnt) for k, v in self.cur_vocab_subword_freqs.items()}
            
               
    def train_tokenizer(self, corpus):
        self.train_initial_tokenizer(corpus)
        
        while len(self.cur_vocab_subword_freqs.keys()) > self.vocab_size:
            
            viterbi_word_freq = self.cur_vocab_subword_freqs
            viterbi_logprob = self.cur_vocab_subword_logprob
            
            for i in range(self.sub_em_steps):
                viterbi_word_freq, _ = self.run_e_step(viterbi_logprob)
                viterbi_logprob = self.run_m_step(viterbi_word_freq)
            viterbi_losses = []

            for token in self.cur_vocab_subword_freqs:  
                _, delta = self.delta_loss(token, viterbi_word_freq, viterbi_logprob)
                viterbi_losses.append((token, delta))
                        
            viterbi_losses = sorted(viterbi_losses, key=lambda x: x[1])
            
            viterbi_losses = viterbi_losses[:max(int(len(viterbi_losses)*(1. - self.shrink_multplier)), self.vocab_size)]
            tokens = list(map(lambda x: x[0], viterbi_losses))
            tokens = set(tokens).union(set(self.alphabet))
            tokens = list(tokens)
            
            self.rebuid_vocab(tokens)
            
            if len(viterbi_losses) == self.vocab_size:
                break
            
        for i, key in enumerate(self.cur_vocab_subword_logprob.keys()):
            self.stoi[key] = i
            self.itos[i] = key
            
        self.vocab_size = len(self.cur_vocab_subword_logprob)
        
    def tokenize(self, text):
        words = self.pre_tokinze_str(text)
        tokens = []
        
        for word in words:
            cur_token, _, _ = self.get_viterbi_path(word, self.cur_vocab_subword_logprob)
            tokens.extend(cur_token)
        
        return tokens
    
    def encode(self, text):
        tokens = self.tokenize(text)
        encoded = []
        for t in tokens:
            encoded.append(self.stoi[t])
            
        return encoded
    
    def decode(self, enc):
        tokens = []
        for i in enc:
            tokens.append(self.itos[i])
        
        return re.sub(self.initial_tokenizer.special_symbol, ' ', ''.join(tokens))