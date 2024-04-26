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
for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix] # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)

g = torch.Generator().manual_seed(2147483647) # seed the random number generator
C = torch.randn((27,2), generator= g) # initialize the character embeddings randomly
W1 = torch.randn((6,100), generator= g) # initialize the weights randomly
b1 = torch.randn(100, generator= g) # initialize the bias randomly
W2 = torch.randn((100,27), generator= g) # initialize the weights randomly
b2 = torch.randn(27, generator= g) # initialize the bias randomly
parameters = [C, W1, b1, W2, b2] # collect all parameters

for p in parameters:
    p.requires_grad = True # setting requires_grad=True

learningStepExponent = torch.linspace(-3, 0, 1000)
learningStep = 10**learningStepExponent
lri = []
lossi = []

for i in range(1000): # train for 1000 iterations
    # minibatch construction
    ix = torch.randint(0, X.shape[0], (32,)) # random indices

    # forward pass
    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1,6) @ W1 + b1) # compute hidden states
    logits = h @ W2 + b2 # compute logits
    #counts = logits.exp() # compute the softmax
    #prob = counts / counts.sum(-1, keepdims=True) # normalize to get a probability
    #loss = -prob[torch.arange(32), Y].log().mean() # cross-entropy loss
    loss = F.cross_entropy(logits, Y[ix])
    print(loss.item())

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward() # autograd computes all the gradients

    # update
    lr = learningStep[i]
    for p in parameters:
        p.data +=(-lr * p.grad) # perform a GD step

# print(loss.item())