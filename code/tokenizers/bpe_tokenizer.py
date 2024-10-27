from collections import defaultdict

import nltk
import itertools

import tqdm
import numpy as np
import re


class BPETokenizer():
    def __init__(self, vocab_size):
        self.word_freq = defaultdict(int)
        self.vocab = []
        self.vocab_freqs = defaultdict(int)
        self.merges = {}
        
        self.smallest_vocab = []
        
        self.vocab_size = vocab_size
        self.special_symbol = "Ġ"
        
        self.stoi_vocab = {}
        self.itos_vocab = {}
        
    def pre_tokenize_str(self, text):
        text_tokenized_with_spaces = [[[' '] + nltk.word_tokenize(w)] if idx != 0 else [nltk.word_tokenize(w)]  for idx, w in enumerate(text.split(' '))]
        text_tokenized_with_spaces = list(itertools.chain(*list(itertools.chain(*text_tokenized_with_spaces))))
        
        for i in range(len(text_tokenized_with_spaces)):
            if text_tokenized_with_spaces[i] == ' ':
                text_tokenized_with_spaces[i] = self.special_symbol
                
        tokenized_text = []
        i = 0
         
        while i < len(text_tokenized_with_spaces):
            if i < len(text_tokenized_with_spaces) - 1:
                if text_tokenized_with_spaces[i] == self.special_symbol and text_tokenized_with_spaces[i + 1] != self.special_symbol:
                    tokenized_text.append(self.special_symbol + text_tokenized_with_spaces[i + 1])
                    i += 2
                else:
                    tokenized_text.append(text_tokenized_with_spaces[i])
                    i += 1
            else:
                tokenized_text.append(text_tokenized_with_spaces[i])
                i += 1
                
        return tokenized_text
    
    def pre_tokenize_text(self, text):
        strings = text.split('\n')
        
        res = []
        for i, str in enumerate(strings):
            if str == '' and i == 0:
                res.append('\n')
            else:
                pre_tokenize = self.pre_tokenize_str(str)
                res.extend(pre_tokenize)
                if i != len(strings) - 1: 
                    res.append('\n')
        
        return res    
            
    def compute_pair_freqs(self, splits):
        pair_freqs = defaultdict(lambda : 0)
        
        for word, freq in self.word_freq.items():
            split = splits[word]
            
            if len(split) == 1:
                continue
            
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                
                pair_freqs[pair] += freq
                
        return pair_freqs
    
    def merge_pair(self, a, b, splits):
        for word in self.word_freq:
            split = splits[word]
            
            if len(split) == 1:
                continue
            
            i = 0
            
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
                    
            splits[word] = split
            
        return splits
    
    def train_tokenizer(self, corpus):
        for text in tqdm.tqdm(corpus):
            words = self.pre_tokenize_text(text)
            
            for word in words:
                self.word_freq[word] += 1
            
        alphabet = set()
        
        for word in self.word_freq.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.add(letter)
                self.vocab_freqs[letter] += 1
        
        alphabet = list(alphabet) + ['\n']
        alphabet.sort()    
        self.vocab = alphabet#["<|endoftext|>"] + alphabet
        self.smallest_vocab = self.vocab.copy()
        
        splits = {word : [c for c in word] for word in self.word_freq.keys()}
        
        prev_vocab_len = len(self.vocab)
        
        pbar = tqdm.tqdm(total=self.vocab_size)
        pbar.update(prev_vocab_len)
        
        while len(self.vocab) < self.vocab_size:
            pbar.update(len(self.vocab) - prev_vocab_len)
            prev_vocab_len = len(self.vocab)
        
            pair_freqs = self.compute_pair_freqs(splits)
        
            max_freq = None
            best_pair = ''
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
        
            self.merges[best_pair] = ''.join(best_pair)
            self.vocab_freqs[''.join(best_pair)] = max_freq
            splits = self.merge_pair(*best_pair, splits)
            
            self.vocab.append(best_pair[0] + best_pair[1])
            
        for i, s in enumerate(self.vocab):
            self.stoi_vocab[s] = i
            self.itos_vocab[i] = s
            
        self.vocab_size = len(self.vocab)
            
    
    def tokenize(self, text):
        pre_tokenized_text = self.pre_tokenize_text(text)
        
        splits = [[l for l in word] for word in pre_tokenized_text]
                
        for pair, merge in tqdm.tqdm(self.merges.items()):
            i = 0
            
            for idx, split in enumerate(splits):
                i = 0
                
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split
        res = []
        
        for s in splits:
            res.extend(s) 
        return res
    
    def encode(self, text):
        tokeneized = self.tokenize(text)
        encoded = []
        
        for t in tokeneized:
            encoded.append(self.stoi_vocab[t])
            
        return encoded
    
    def decode(self, idxs):
        decoded = []
        
        for l in idxs:
            decoded.append(self.itos_vocab[l])
            
        decoded = ''.join(decoded)
        return re.sub(self.special_symbol, ' ', decoded)
        