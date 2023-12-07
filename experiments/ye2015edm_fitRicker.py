import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
class Ricker(nn.Module):
    """
     The Ricker for Spwaner-Recruit relationship as in Ye,.. Sugihara.
    """
    def __init__(self, params):

        super().__init__()

        self.model_params = torch.nn.Parameter(torch.tensor(params, requires_grad=True, dtype=torch.double))
    def forward(self, x, p4, recursive = False):

        S, U = x[:,:,0], x[:,:,1]
        alpha, beta, d, g = self.model_params

        if recursive:
            Rhat = torch.full_like(S, torch.nan)
            Ret = torch.full_like(S, torch.nan)
            Rhat[:, 0] = S[:, 0].clone()
            for i in range(1,Rhat.shape[1]):
                Rhat[:, i] = Rhat[:, i-1] * torch.exp(alpha - beta * Rhat[:, i-1] + d * U[:, i] + g * (U[:, i] ** 2))
                Ret[:, i] = Rhat[:, i] * p4 + Rhat[:, i - 1] * p4
        else:
            Rhat = torch.full_like(S, torch.nan)
            for i in range(Rhat.shape[1]):
                Rhat[:, i] = S[:, i] * torch.exp(alpha - beta * S[:, i] + d * U[:, i] + g * (U[:, i] ** 2))
            Ret = torch.full_like(S, torch.nan)
            for i in range(1, Ret.shape[1]):
                Ret[:,i] = Rhat[:,i]*p4 + Rhat[:,i-1]*p4

        return Ret

    def __repr__(self):
        return f" alpha: {self.model_params[0].item()}, \
            beta: {self.model_params[1].item()}, \
                d: {self.model_params[2].item()}, \
        g: {self.model_params[3].item()}, \
                    "

class SimData(Dataset):
    def __init__(self, X, y, seqlen = 2):
        self.X = X
        self.y = y
        self.seqlen = seqlen
    def __len__(self):
        return self.X.shape[0] - (self.seqlen - 1)
    def __getitem__(self, index: int):
        return self.X[index:(index+self.seqlen),:], self.y[index:(index+self.seqlen),:]

def plot_losses(losses, saveto=''):

    ll = torch.stack(losses).detach().numpy()
    plt.plot(ll)
    plt.ylabel(f'MSE loss')
    plt.xlabel(f'Epoch')
    #plt.savefig(os.path.join(saveto, 'losses.pdf'))
    plt.show()


# Load the salmon tibble
for i in range(9):
    data = pd.read_csv(f'data/tibble_{i+1}.csv')
    if data['stk'][0] == 'Seymour':
        break
data['ET_apr'].plot()
data['PDO_win'].plot()
data['eff'].plot() # spawners
data['rec45'].plot() # recruits
data['ret'].plot() # returns
p4 = (data['rec4']/data['rec45']).mean()

X = np.array([data['eff'].to_numpy(), data['PDO_win'].to_numpy()])
y = data['ret'].to_numpy()[np.newaxis, :]
mask = np.isnan(y)
mask[:,0] = False
X, y = np.transpose(X[:,~mask.squeeze()]), np.transpose(y[~mask][np.newaxis, :])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

X_train, y_train = torch.tensor(X_train, dtype=torch.float, requires_grad=True), torch.tensor(y_train, dtype=torch.float, requires_grad=True)
X_test, y_test = torch.tensor(X_test, dtype=torch.float, requires_grad=True), torch.tensor(y_test, dtype=torch.float, requires_grad=True)

#===========================#
# Fit model non-recursively #
#===========================#

dat = SimData(X = X_train, y = y_train)
trainloader = DataLoader(dat, batch_size=16, shuffle=False, drop_last=False)
for b in trainloader:
    b1, b2 = b

model = Ricker([1.8, 1.1, 0.9, 0.9])
optimizer = torch.optim.Adam([{'params':model.model_params}], lr=0.0001)
criterion = torch.nn.MSELoss()
losses = []
for epoch in range(1500):
    if epoch % 50 == 0:
        print('Epoch:', epoch)
    for b in trainloader:
        b1, b2 = b
        out = model.forward(b1, p4=p4, recursive=True)
        #out = out[:,-1].view(16, 1)
        loss = criterion(out[:,1], b2.squeeze()[:,1])
        loss.backward()
        losses.append(loss.clone())
        optimizer.step()

ll = torch.stack(losses).detach().numpy()
plt.plot(ll)
model.model_params

preds_train = model(X_train.view(1,X_train.shape[0], X_train.shape[1]), p4 = p4).detach().numpy()
preds_test = model(X_test.view(1,X_test.shape[0], X_test.shape[1]), p4 = p4).detach().numpy()

plt.plot(np.transpose(preds_train))
plt.plot(y_train.detach().numpy())

plt.plot(np.transpose(preds_test))
plt.plot(y_test.detach().numpy())
plt.hlines(y_train.detach().numpy() [~np.isnan(y_train.detach().numpy() )].mean(), xmin = 0, xmax=12)

#=======================#
# Fit model recursively #
#=======================#

dat = SimData(X = X_train, y = y_train, seqlen=15)
trainloader = DataLoader(dat, batch_size=8, shuffle=False, drop_last=True)
for b in trainloader:
    b1, b2 = b

model = Ricker([.5, 0.6, 0.7, 0.7])
optimizer = torch.optim.Adam([{'params':model.model_params}], lr=0.0001)
criterion = torch.nn.MSELoss()
losses = []
for epoch in range(5000):
    if epoch % 50 == 0:
        print('Epoch:', epoch)
    for b in trainloader:
        b1, b2 = b
        out = model.forward(b1, p4=p4)
        #out = out[:,-1].view(16, 1)
        loss = criterion(out[:,1], b2.squeeze()[:,1])
        loss.backward()
        losses.append(loss.clone())
        optimizer.step()

ll = torch.stack(losses).detach().numpy()
plt.plot(ll)
model.model_params

preds_train = model(X_train.view(1,X_train.shape[0], X_train.shape[1]), p4 = p4, recursive = True).detach().numpy()
preds_test = model(X_test.view(1,X_test.shape[0], X_test.shape[1]), p4 = p4, recursive = True).detach().numpy()

plt.plot(np.transpose(preds_train))
plt.plot(y_train.detach().numpy())

plt.plot(np.transpose(preds_test))
plt.plot(y_test.detach().numpy())
plt.hlines(y_train.detach().numpy() [~np.isnan(y_train.detach().numpy() )].mean(), xmin = 0, xmax=12)
