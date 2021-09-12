import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message='source.*has changed')
import torch
import torch.nn.functional as F
import os

from transformers import *


class Vocabulary:
    '''
    Shared vocabulary object for LSTM and GPT2
    '''
    def __init__(self, model_type, tokenizer = None, path = None):
        self.type = model_type

        if self.type == 'LSTM':
            vocab_path = os.path.join(path, 'vocab.txt')
            vocab = open(vocab_path, encoding="utf8").read()
            self.idx2word = [w for w in vocab.split()]
            self.transformer = False
        else:
            self.tokenizer = tokenizer
            self.transformer = True
            self.idx2word = list(tokenizer.encoder.keys())
        self.word2idx = {self.idx2word[i]: i for i in range(len(self.idx2word))}
        self.size = len(self.idx2word)
    def encode(self, text):
        if self.transformer:
            return self.tokenizer.encode(text)
        else:
            text = text.split()
            data = []
            for token in range(len(text)):
                word = text[token]
                if word not in self.word2idx:
                    data.append(self.word2idx["<unk>"])
                else:
                    data.append(self.word2idx[word])
            return data

    def decode(self, text_ids, skip_special_tokens = True):
        if self.transformer:
            text_ids = list(text_ids)
            text_ids = [int(i) for i in text_ids]
            decoded = self.tokenizer.decode(text_ids, skip_special_tokens = skip_special_tokens)
            return decoded
        else:
            decoded = [self.idx2word[int(i)] for i in text_ids]
            if skip_special_tokens:
                if '<eos>' in decoded: decoded.remove( '<eos>' )
            return ' '.join(decoded)



def load_lm(model_type = 'LSTM', dir_path = 'LSTM/', cuda = False):
    device = torch.device("cuda" if cuda else "cpu")
    if model_type == 'LSTM':
        import model
        import utils
        lm = torch.load(dir_path + 'model.pt', map_location=device)
        lm_vocabulary = Vocabulary(model_type, path = dir_path )
    elif model_type == 'distilGPT2':
        lm_vocabulary = Vocabulary(model_type, tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large', add_prefix_space = True))
        lm = GPT2LMHeadModel.from_pretrained('distilgpt2')
    elif model_type == 'GPT2':
        lm_vocabulary = Vocabulary(model_type, tokenizer=GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space = True) )
        lm = GPT2LMHeadModel.from_pretrained('gpt2')
    if cuda: lm.cuda()
    lm.eval()
    return lm, lm_vocabulary

