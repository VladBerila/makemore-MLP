import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
import pathlib # for dealing with file paths
import random # for shuffling the data

path = pathlib.Path().resolve() # get the path to the current directory

block_size = 3 # context length: how many characters do we take to predict the next one?

#read in all the words
words = open(path.joinpath('names.txt'), 'r').read().splitlines()

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# build the dataset
def build_dataset(words):
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
    return X, Y

# 80% training set, 10% dev/validation set, 10% test set
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
random.seed(42) # seed the random number generator
random.shuffle(words)

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647) # seed the random number generator
C = torch.randn((27,10), generator= g) # initialize the character embeddings randomly
W1 = torch.randn((30,200), generator= g) # initialize the weights randomly
b1 = torch.randn(200, generator= g) # initialize the bias randomly
W2 = torch.randn((200,27), generator= g) # initialize the weights randomly
b2 = torch.randn(27, generator= g) # initialize the bias randomly
parameters = [C, W1, b1, W2, b2] # collect all parameters


def train_set(params, numIterations, learningStep):
    for p in params:
        p.requires_grad = True # setting requires_grad=True

    #learningStepExponent = torch.linspace(-3, 0, 1000)
    #learningStep = 10**learningStepExponent
    #lri = []
    lossi = []
    stepi = []

    for i in range(numIterations): # train for numIterations iterations
        # minibatch construction
        ix = torch.randint(0, Xtr.shape[0], (32,)) # random indices

        # forward pass
        emb = C[Xtr[ix]]
        h = torch.tanh(emb.view(-1,30) @ W1 + b1) # compute hidden states
        logits = h @ W2 + b2 # compute logits
        #counts = logits.exp() # compute the softmax
        #prob = counts / counts.sum(-1, keepdims=True) # normalize to get a probability
        #loss = -prob[torch.arange(32), Y].log().mean() # cross-entropy loss
        loss = F.cross_entropy(logits, Ytr[ix])

        # backward pass
        for p in params:
            p.grad = None
        loss.backward() # autograd computes all the gradients

        # update
        lr = learningStep
        for p in params:
            p.data +=(-lr * p.grad) # perform a GradientDescent step

        # track status
        # lri.append(learningStepExponent[i])
        # lossi.append(loss.log10().item())
        # stepi.append(i)

    #plt.plot(stepi, lossi)
    #plt.show(block=True)

    return params, loss

parameters, loss = train_set(parameters, 100000, 0.1)
parameters, loss = train_set(parameters, 100000, 0.01)
#parameters, loss = train_set(parameters, 60000, 0.01)
print(loss)

# check the whole training set(which was sampled)
emb = parameters[0][Xtr]
h = torch.tanh(emb.view(-1,30) @ parameters[1] + parameters[2])
logits = h @ parameters[3] + parameters[4]
loss = F.cross_entropy(logits, Ytr)
print(loss)

#check the dev set
emb = parameters[0][Xdev]
h = torch.tanh(emb.view(-1,30) @ parameters[1] + parameters[2])
logits = h @ parameters[3] + parameters[4]
loss = F.cross_entropy(logits, Ydev)
print(loss)

# evaluate the model
#plt.figure(figsize=(8, 8))
#plt.scatter(parameters[0][:,0].data, parameters[0][:,1].data, s=200)
#for i in range(parameters[0].shape[0]):
#    plt.text(parameters[0][i,0].item(), parameters[0][i,1].item(), itos[i], ha="center", va = "center", color='white')
#plt.grid('minor')
#plt.show(block=True)

# sample from the model
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = parameters[0][torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ parameters[1] + parameters[2])
        logits = h @ parameters[3] + parameters[4]
        prob = F.softmax(logits, dim=1)
        next = torch.multinomial(prob, num_samples=1)
        context = context[1:] +[next]
        out.append(next.item())
        if next.item() == 0:
            break
    print(''.join([itos[i] for i in out]))