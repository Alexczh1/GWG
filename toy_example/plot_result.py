import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device(0 if torch.cuda.is_available() else 'cpu')

seed = 1234
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

clearity=0.07

fig = plt.figure()
ax = fig.add_subplot()

# plot Figure 2

kl_list_mean = []

with open(f'gd_result1.5old_2d_100particle_kl_mean.txt', 'r') as f:
    for line in f:
        kl_list_mean.append(float(line))
p=kl_list_mean.pop(-1)
ax.plot(np.linspace(0, 20000, num=len(kl_list_mean)), kl_list_mean, label='Ada-GWG(p$_{0}$=1.5)',color='g',linestyle='-')
kl_list_upper = []
kl_list_lower = []

with open(f'gd_result1.5old_2d_100particle_kl_upper.txt', 'r') as f:
    for line in f:
        kl_list_upper.append(float(line))
p=kl_list_upper.pop(-1)

with open(f'gd_result1.5old_2d_100particle_kl_lower.txt', 'r') as f:
    for line in f:
        kl_list_lower.append(float(line))
p=kl_list_lower.pop(-1)

x=np.linspace(0, 20000, num=len(kl_list_upper))
y1=np.array(kl_list_upper)
y2=np.array(kl_list_lower)

ax.fill_between(x, y1, y2,alpha=clearity+0.1,facecolor='g', where=y1 >= y2,  interpolate=True)
ax.fill_between(x, y1, y2, alpha=clearity+0.1,facecolor='g',where=y1 <= y2,  interpolate=True)

#####################################################################




kl_list_mean = []

with open(f'gd_result2.0old_2d_100particle_kl_mean.txt', 'r') as f:
    for line in f:
        kl_list_mean.append(float(line))
p=kl_list_mean.pop(-1)
ax.plot(np.linspace(0, 20000, num=len(kl_list_mean)), kl_list_mean, label='Ada-GWG(p$_{0}$=2.0)',color='r',linestyle='-')



kl_list_upper = []
kl_list_lower = []

with open(f'gd_result2.0old_2d_100particle_kl_upper.txt', 'r') as f:
    for line in f:
        kl_list_upper.append(float(line))
p=kl_list_upper.pop(-1)

with open(f'gd_result2.0old_2d_100particle_kl_lower.txt', 'r') as f:
    for line in f:
        kl_list_lower.append(float(line))
p=kl_list_lower.pop(-1)

x=np.linspace(0, 20000, num=len(kl_list_upper))

y1=np.array(kl_list_upper)
y2=np.array(kl_list_lower)

ax.fill_between(x, y1, y2,alpha=clearity,facecolor='r', where=y1 >= y2,  interpolate=True)
ax.fill_between(x, y1, y2, alpha=clearity,facecolor='r',where=y1 <= y2,  interpolate=True)






#####################################################################




kl_list_mean = []

with open(f'gd_result2.2old_2d_100particle_kl_mean.txt', 'r') as f:
    for line in f:
        kl_list_mean.append(float(line))
p=kl_list_mean.pop(-1)
ax.plot(np.linspace(0, 20000, num=len(kl_list_mean)), kl_list_mean, label='Ada-GWG(p$_{0}$=2.2)',color='c',linestyle='-')



kl_list_upper = []
kl_list_lower = []

with open(f'gd_result2.2old_2d_100particle_kl_upper.txt', 'r') as f:
    for line in f:
        kl_list_upper.append(float(line))
p=kl_list_upper.pop(-1)

with open(f'gd_result2.2old_2d_100particle_kl_lower.txt', 'r') as f:
    for line in f:
        kl_list_lower.append(float(line))
p=kl_list_lower.pop(-1)

x=np.linspace(0, 20000, num=len(kl_list_upper))

y1=np.array(kl_list_upper)
y2=np.array(kl_list_lower)

ax.fill_between(x, y1, y2,alpha=clearity,facecolor='c', where=y1 >= y2,  interpolate=True)
ax.fill_between(x, y1, y2, alpha=clearity,facecolor='c',where=y1 <= y2,  interpolate=True)






#####################################################################




kl_list_mean = []

with open(f'gd_result1.5old_2d_fixed_100particle_kl_mean.txt', 'r') as f:
    for line in f:
        kl_list_mean.append(float(line))
p=kl_list_mean.pop(-1)
ax.plot(np.linspace(0, 20000, num=len(kl_list_mean)), kl_list_mean, label='GWG(p=1.5)',color='g',linestyle=':')



kl_list_upper = []
kl_list_lower = []

with open(f'gd_result1.5old_2d_fixed_100particle_kl_upper.txt', 'r') as f:
    for line in f:
        kl_list_upper.append(float(line))
p=kl_list_upper.pop(-1)

with open(f'gd_result1.5old_2d_fixed_100particle_kl_lower.txt', 'r') as f:
    for line in f:
        kl_list_lower.append(float(line))
p=kl_list_lower.pop(-1)

x=np.linspace(0, 20000, num=len(kl_list_upper))

y1=np.array(kl_list_upper)
y2=np.array(kl_list_lower)

ax.fill_between(x, y1, y2,alpha=clearity,facecolor='g', where=y1 >= y2,  interpolate=True)
ax.fill_between(x, y1, y2, alpha=clearity,facecolor='g',where=y1 <= y2,  interpolate=True)




#####################################################################




kl_list_mean = []

with open(f'gd_result2.0old_2d_fixed_100particle_kl_mean.txt', 'r') as f:
    for line in f:
        kl_list_mean.append(float(line))
p=kl_list_mean.pop(-1)
ax.plot(np.linspace(0, 20000, num=len(kl_list_mean)), kl_list_mean, label='GWG(p=2.0)',color='r',linestyle=':')



kl_list_upper = []
kl_list_lower = []

with open(f'gd_result2.0old_2d_fixed_100particle_kl_upper.txt', 'r') as f:
    for line in f:
        kl_list_upper.append(float(line))
p=kl_list_upper.pop(-1)

with open(f'gd_result2.0old_2d_fixed_100particle_kl_lower.txt', 'r') as f:
    for line in f:
        kl_list_lower.append(float(line))
p=kl_list_lower.pop(-1)

x=np.linspace(0, 20000, num=len(kl_list_upper))

y1=np.array(kl_list_upper)
y2=np.array(kl_list_lower)

ax.fill_between(x, y1, y2,alpha=clearity,facecolor='r', where=y1 >= y2,  interpolate=True)
ax.fill_between(x, y1, y2, alpha=clearity,facecolor='r',where=y1 <= y2,  interpolate=True)




#####################################################################




kl_list_mean = []

with open(f'gd_result2.2old_2d_fixed_100particle_kl_mean.txt', 'r') as f:
    for line in f:
        kl_list_mean.append(float(line))
p=kl_list_mean.pop(-1)
ax.plot(np.linspace(0, 20000, num=len(kl_list_mean)), kl_list_mean, label='GWG(p=2.2)',color='c',linestyle=':')



kl_list_upper = []
kl_list_lower = []

with open(f'gd_result2.2old_2d_fixed_100particle_kl_upper.txt', 'r') as f:
    for line in f:
        kl_list_upper.append(float(line))
p=kl_list_upper.pop(-1)

with open(f'gd_result2.2old_2d_fixed_100particle_kl_lower.txt', 'r') as f:
    for line in f:
        kl_list_lower.append(float(line))
p=kl_list_lower.pop(-1)

x=np.linspace(0, 20000, num=len(kl_list_upper))

y1=np.array(kl_list_upper)
y2=np.array(kl_list_lower)

ax.fill_between(x, y1, y2,alpha=clearity,facecolor='c', where=y1 >= y2,  interpolate=True)
ax.fill_between(x, y1, y2, alpha=clearity,facecolor='c',where=y1 <= y2,  interpolate=True)



ax.legend()
plt.legend(loc="upper right")
plt.savefig('adam.jpg', dpi=600)
plt.show()



