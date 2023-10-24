import os
import math
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Distribution

device = torch.device(0 if torch.cuda.is_available() else 'cpu')
import time

import matplotlib.pylab as plt
import torch.nn as nn
from torch import autograd
from tqdm import trange
import ite
from torch.distributions.normal import Normal
from torch.distributions.studentT import StudentT
from torch.distributions.gamma import Gamma
import torch.nn.functional as F
from scipy.stats import laplace, multivariate_t, cauchy, gamma, cramervonmises

import random
import seaborn as sns


# seed = 1224
# seed = 1234
# seed = 1244
seed = 1254

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class EXP(nn.Module):

    def __init__(self, dim, log_alpha_init=0.7, beta_init=1.0):
        super().__init__()
        self.log_alpha = nn.Parameter(
            torch.tensor(log_alpha_init, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(beta_init, requires_grad=True))
        self.dim = dim

    def forward(self, x):
        alpha = self.log_alpha.exp()
        score = -alpha * self.beta * x.abs()**(alpha - 1) * x.sgn()
        return score

    def log_prob(self, x):
        alpha = self.log_alpha.exp()
        logp = torch.log(self.beta) / alpha + torch.log(alpha) - torch.lgamma(
            1 / alpha) - self.beta * x.abs()**alpha - np.log(2)
        if self.dim > 1:
            logp = logp.sum(dim=1)
        return logp

    def sampler(self, n_sample):
        alpha = self.log_alpha.exp()
        x0 = Gamma(alpha, rate=torch.tensor(1.0)).sample((n_sample, self.dim))

        # ind = torch.tensor(np.random.randint(0, 2, n_sample) * 2 - 1).view(
        #     n_sample, 1).expand_as(x0) #half

        ind = torch.tensor(np.random.randint(0, 2, (n_sample,self.dim)) * 2 - 1).expand_as(x0)

        x = ind * (x0 / self.beta)**(1 / alpha)
        return x

exp=EXP(dim = 2, log_alpha_init=-0.1, beta_init=0.3)
sample = exp.sampler(n_sample=100).cpu().detach().numpy()
# plt.plot(sample[:,0],sample[:,1],'.')
# plt.show()

##################################################################################

def divergence_bf(dx, y):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()
def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1
def sample_gaussian_like(y):
    return torch.randn_like(y)
def divergence_approx(f, y, e=None):
    if e is None:
        e = sample_rademacher_like(y)
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx
    
################################################################################
    
class PFGfast:
  def __init__(self, P, net, optimizer1,optimizer2):
    self.P = P
    self.net = net
    self.optim1 = optimizer1
    self.optim2 = optimizer2
    self.p_update=p_update

  def phi(self, X):
    phi = self.net(X) / X.size(0)
    return -phi

  def step(self, X):
    self.optim1.zero_grad()
    X.grad = self.phi(X)
    self.optim1.step()

  def score_step(self, X, p_norm):
    H1 = torch.std(X,0)
    H1 = 1.0 / H1
    H1=torch.pow(H1, 0.2)
    H=torch.diag(H1)

    X= X.detach().cpu().requires_grad_(True)
    log_prob = self.P.log_prob(X)
    score_func = torch.autograd.grad(log_prob.sum(), X)[0].detach()
    
    self.net.train()
    X = X.to(device)

    S = self.net(X)
    self.optim2.zero_grad()
    score_func = score_func.to(device)

    # H_2
    loss = (-torch.sum(score_func * S) - torch.sum(divergence_approx(S, X)) + 0.5 * torch.trace(
        H.matmul(S.T).matmul(S))) / S.shape[0]

    # lp
    # score_func=score_func.to(device)
    # loss = (-torch.sum(score_func * S) - torch.sum(divergence_approx(S, X)) + torch.norm(S, p=p_norm)**p_norm /p_norm   )/ S.shape[0]

    ##################################################################################################################################################
    # leads to dissatisfying results:
    # l2p
    # loss = (-torch.sum(score_func * S) - torch.sum(divergence_approx(S, X)) + torch.norm(S,p=2) ** p_norm / p_norm) / S.shape[0]
    # exp
    # loss = (-torch.sum(score_func * S) - torch.sum(divergence_approx(S, X)) + torch.exp(torch.norm(S, p=p_norm) ** p_norm )) / S.shape[0]
    # -exp
    # loss = (-torch.sum(score_func * S) - torch.sum(divergence_approx(S, X)) - torch.exp(-(torch.norm(S, p=p_norm) ** p_norm) )) / S.shape[0]
    # ln
    # loss = (-torch.sum(score_func * S) - torch.sum(divergence_approx(S, X)) + torch.norm(S, p=p_norm) ** p_norm * torch.log(torch.norm(S, p=p_norm) ** p_norm ) - torch.exp( (torch.log(torch.norm(S, p=p_norm) ** p_norm ))).abs() ) / S.shape[0]

    loss.backward()
    self.optim2.step()
    scoredifference = torch.abs(S) ** p_norm

    log_scoredifference = torch.log(1 / (torch.abs(S) ** (p_norm - 1)))
    stepsize = 1
    p_update=stepsize * ((1 / p_norm ** 2) * torch.sum(scoredifference) - (1 / ((p_norm - 1) ** 2 * p_norm)) * torch.sum(scoredifference * log_scoredifference)) / S.shape[0]
    # p_adam
    # self.p_update=stepsize *((1 / p_norm ** 2) * torch.sum(scoredifference) - (1 / ((p_norm - 1) ** 2 * p_norm)) * torch.sum(scoredifference * log_scoredifference)) / S.shape[0]

    if p_norm - p_update > 1.1:
        p_norm -= p_update
        p_norm = p_norm.item()
        # print(1)
        return 1, p_norm

    else:
        # print(0)
        return 0, p_norm

  def kl_distance(self):
        KL = ite.cost.BDKL_KnnK()
        x_sample = X.detach().cpu().numpy()
        x_true = self.P.sampler(n_sample=100).cpu().detach().numpy()
        kl = KL.estimation(x_sample, x_true)
        return kl


###########################################################################

from torch import nn
n = 1000
t1 = time.time()
check_frq=100

X_0 = torch.randn(n, 2)
X_0 = X_0.to(device)

h = 32
net = nn.Sequential(
    nn.Linear(2, h),
    nn.Tanh(),
    nn.Linear(h, h),
    nn.Tanh(),
    nn.Linear(h, 2))
torch.save(net.state_dict(), "net.pth")

fig = plt.figure()
ax = fig.add_subplot()


# p_list=[1.5,2.0,2.2]
p_list=[2.2]

Epoch=20000

for p in p_list:
    X=X_0.clone()
    net.load_state_dict(torch.load("net.pth"))
    net = net.to(device)

    optim1 = torch.optim.Adam([X], lr=0.001)
    optim2 = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, nesterov=1)

    p_update=0

    pfg = PFGfast(exp, net, optim1, optim2)
    p_0=p
    kl_list = np.zeros(Epoch // check_frq + 1)
    kl = pfg.kl_distance()
    kl_list[0] = kl

    for i in range(100):
        pfg.score_step(X, p)
    count=0

    countlist = np.zeros(3)

    for i in range(Epoch):
        for j in range(5):
            if i > 5000: #adaptive p
            # if i > 50000: #non-adaptive p
                flag, p_nm = pfg.score_step(X, p)
                # print(flag)
                # print(p_nm)
                p=p_nm

            else:
                flag, p_nm=pfg.score_step(X, p)

        pfg.step(X)

        if (i + 1) % check_frq == 0:
            kl = pfg.kl_distance()
            kl_list[(i + 1) //
                    check_frq] = kl

    print(kl_list)
    print(count)
    print(p)
    print(countlist)

    f = open(f'gd_result{p_0}old_2d_fixed_100particle_kl_seed_{seed}.txt', 'w')
    for d in kl_list:
        f.write(str(d)+"\n")
    f.write(str(p) + "\n")
    f.close()

    ax.plot(np.linspace(0, Epoch, num=len(kl_list)), kl_list, label=f'{p_0:.2f} to {p:.2f}')


ax.legend()
plt.savefig('particle_vi_mog_kl.jpg', dpi=600)
plt.show()

t2 = time.time()
print(f'Computation Time: {t2-t1}')
