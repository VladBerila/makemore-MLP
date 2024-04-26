import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
import pathlib # for dealing with file paths

path = pathlib.Path().resolve() # get the path to the current directory

#read in all the words
words = open(path.joinpath('names.txt'), 'r').read().splitlines()

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# build the dataset

block_size = 3 # context length: how many characters do we take to predict the next one?
X, Y = [], []
for w in words[:5]:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix] # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn((27,2)) # initialize the character embeddings randomly