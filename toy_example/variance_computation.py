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


#computing variances


p=1.5



# kl_list = []
#
# with open(f'gd_result1.1old_2d_100particle_kl.txt', 'r') as f:
#     for line in f:
#         kl_list.append(float(line))
# p=kl_list.pop(-1)
# ax.plot(np.linspace(0, 20000, num=len(kl_list)), kl_list, label=f'{1.1:.2f} to {p:.2f}',color='b',linestyle='-')
#


# #lp
#
# txt = np.loadtxt(f'gd_result{p}old_2d_fixed_100particle_kl_seed_1224.txt')
# txtDF = pd.DataFrame(txt)
#
# txt2 = np.loadtxt(f'gd_result{p}old_2d_fixed_100particle_kl.txt')
# txtDF2 = pd.DataFrame(txt2)
#
# txt3 = np.loadtxt(f'gd_result{p}old_2d_fixed_100particle_kl_seed_1244.txt')
# txtDF3 = pd.DataFrame(txt3)
#
# txt4 = np.loadtxt(f'gd_result{p}old_2d_fixed_100particle_kl_seed_1254.txt')
# txtDF4 = pd.DataFrame(txt4)


#H2
txt = np.loadtxt(f'gd_result_H2_0.2_old_2d_fixed_100particle_kl_seed_1224.txt')
txtDF = pd.DataFrame(txt)

txt2 = np.loadtxt(f'gd_result_H2_0.2_old_2d_fixed_100particle_kl.txt')
txtDF2 = pd.DataFrame(txt2)

txt3 = np.loadtxt(f'gd_result_H2_0.2_old_2d_fixed_100particle_kl_seed_1244.txt')
txtDF3 = pd.DataFrame(txt3)

txt4 = np.loadtxt(f'gd_result_H2_0.2_old_2d_fixed_100particle_kl_seed_1254.txt')
txtDF4 = pd.DataFrame(txt4)


# computing variances
mse1=txtDF.values.flatten()
mse2=txtDF2.values.flatten()
mse3=txtDF3.values.flatten()
mse4=txtDF4.values.flatten()

mean=(mse1+mse2+mse3+mse4)/4.0
std=np.sqrt((np.square(mse1-mean)+np.square(mse2-mean)+np.square(mse3-mean)+np.square(mse4-mean))/3.0)
upper=mean+std
lower=mean-std

#lp
# np.savetxt(f"gd_result{p}old_2d_fixed_100particle_kl_mean.txt",mean)
#
# np.savetxt(f"gd_result{p}old_2d_fixed_100particle_kl_upper.txt",upper)
#
# np.savetxt(f"gd_result{p}old_2d_fixed_100particle_kl_lower.txt",lower)

#H2
np.savetxt(f"gd_result_H2_0.2_old_2d_fixed_100particle_kl_mean.txt",mean)

np.savetxt(f"gd_result_H2_0.2_old_2d_fixed_100particle_kl_upper.txt",upper)

np.savetxt(f"gd_result_H2_0.2_old_2d_fixed_100particle_kl_lower.txt",lower)
