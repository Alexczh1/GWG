import argparse
import logging
import os
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
import yaml
from models.model import F_net
from models.bnn_models import BNN


class GWGrunner(object):
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.p_norm = torch.tensor([config.model.p_norm], requires_grad = True, device = self.device)
        self.permutation_path = config.permutation_path
    def get_datasets(self, dataset, test_size=0.1):
        if dataset == 'boston':
            data = np.loadtxt('data/boston_housing.txt')
        X = torch.from_numpy(data[:, :-1]).float().to(self.device)
        y = torch.from_numpy(data[:, -1]).float().to(self.device)
        y = y.unsqueeze(1)

        size_train = int(np.round(X.shape[0] * (1-test_size)))
        permutation = torch.load(self.permutation_path)
        index_train = permutation[0: size_train]
        index_test = permutation[size_train:]

        X_train, y_train = X[index_train, :], y[index_train]
        X_test, y_test = X[index_test, :], y[index_test]

        size_dev = min(int(np.round(0.1 * X_train.shape[0])), 500)
        X_train, y_train = X_train[:-size_dev], y_train[:-size_dev]
        X_dev, y_dev = X_train[-size_dev:], y_train[-size_dev:]


        self.x_mean = X_train.mean(0)
        self.x_std = X_train.std(0, unbiased=False)
        X_train = (X_train - self.x_mean) / self.x_std
        X_dev = (X_dev - self.x_mean) / self.x_std
        X_test = (X_test - self.x_mean) / self.x_std
        
        self.y_mean = torch.mean(y_train)
        self.y_std = torch.std(y_train, unbiased=False)

        """normalize y """
        y_train = (y_train - self.y_mean) / (self.y_std)

        return X_train, y_train, X_dev, y_dev, X_test, y_test
    def get_optimizer(self, parameters, p_param = False, optim_ = "Adam"):
        if p_param:
            lr = self.config.optim.p_norm_lr
        else:
            lr = self.config.optim.f_lr
        if optim_ == 'Adam':
            return optim.Adam(parameters,
                              lr=lr,
                              betas=(self.config.optim.beta, 0.999)) 
        elif optim_ == 'RMSProp':
            return optim.RMSprop(parameters,
                                 lr=lr,
                                 weight_decay=self.config.optim.weight_decay)
        elif optim_ == 'SGD':
            return optim.SGD(parameters, lr=lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(
                self.config.optim.optimizer))

    def init_weights(self, X_train, y_train):
        w1 = 1.0 / np.sqrt(X_train.shape[1] + 1) * torch.randn(self.config.training.batch_size, X_train.shape[1] * self.config.model.bnn_latent_dim).to(self.device)
        b1 = torch.zeros((self.config.training.batch_size, self.config.model.bnn_latent_dim)).to(self.device)
        w2 = 1.0 / np.sqrt(self.config.model.bnn_latent_dim + 1) * torch.randn(self.config.training.batch_size, self.config.model.bnn_latent_dim).to(self.device)
        b2 = torch.zeros((self.config.training.batch_size,1)).to(self.device)
        loglambda = torch.ones((self.config.training.batch_size,1)).to(self.device) * np.log(np.random.gamma(1, 0.1))
        loggamma = torch.ones((self.config.training.batch_size,1)).to(self.device)
        for i in range(self.config.training.batch_size):
            with torch.no_grad():
                ridx = np.random.choice(range(X_train.shape[0]), np.min([X_train.shape[0], 1000]), replace = False)
                y_hat = self.model.predict_y(torch.cat([w1[i][None,:],b1[i][None,:],w2[i][None,:],b2[i][None,:]],dim=1).to(self.device), X_train[ridx,:], self.y_mean, self.y_std, scale = False)
                loggamma[i] = -torch.log(torch.mean((y_hat - y_train[ridx,:])**2,1))
        return torch.cat([w1,b1,w2,b2, loggamma, loglambda],dim=1).to(self.device)
    
    def divergence_approx(self, fnet_value, parti_input, e=None):
        def sample_rademacher_like(y):
            return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1
        if e is None:
            e = sample_rademacher_like(parti_input)
        e_dzdx = torch.autograd.grad(fnet_value, parti_input, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = e_dzdx_e.view(parti_input.shape[0], -1).sum(dim=1)
        return approx_tr_dzdx

    def train(self):
        X_train, y_train, X_dev, y_dev, X_test, y_test = self.get_datasets(
            self.config.data.dataset)
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        self.model = BNN(self.device, X_train.shape[1])
        self.scale_sto = X_train.shape[0]/self.config.training.data_batch_size

        self.weight_dim = (input_dim + 1) * self.config.model.bnn_latent_dim + (
            self.config.model.bnn_latent_dim + 1) * output_dim + 2
        f_net = F_net(self.weight_dim,
                      self.config.model.f_latent_dim).to(self.config.device)
        f_optim = self.get_optimizer(f_net.parameters(), optim_ = self.config.optim.f_optimizer)
        scheduler_f = torch.optim.lr_scheduler.StepLR(f_optim, step_size=1000, gamma=0.9)
        p_norm_optim = self.get_optimizer([self.p_norm], p_param=True, optim_ = self.config.optim.p_norm_optimizer)
        rmse_list = []
        ll_list = []
        p_norm_list = []
        auto_corr = 0.9
        fudge_factor = 1e-6
        historical_grad = 0
        self.particle_weight = self.init_weights(X_train, y_train)
        for ep in range(self.config.training.n_epoch):
            N0 = X_train.shape[0]
            batch = [i % N0 for i in range(ep * self.config.training.data_batch_size, (ep + 1) * self.config.training.data_batch_size)]
            x = X_train[batch]
            y = y_train[batch]
            score_target = self.model.score(self.particle_weight, x, y, self.scale_sto)
            for i in range(self.config.training.f_iter):                   
                self.particle_weight.requires_grad_(True)
                f_value = f_net(self.particle_weight)
                f_optim.zero_grad()
                p_norm_item = self.p_norm.item()
                loss = (-torch.sum(score_target * f_value) - torch.sum(self.divergence_approx(f_value, self.particle_weight)) + torch.norm(f_value, p=p_norm_item)**p_norm_item /p_norm_item)/f_value.shape[0]
                loss.backward()
                f_optim.step()
                scheduler_f.step()
                if self.config.optim.p_adaptive:
                    """adaptive update p"""
                    self.p_norm.requires_grad_(True)
                    p_norm_optim.zero_grad()
                    loss_p = - torch.mean(torch.sum((torch.abs(f_value.detach())**(self.p_norm-1))**(self.p_norm/(self.p_norm-1)),dim=-1) /self.p_norm)
                    loss_p.backward()
                    self.p_norm.grad.clamp_(min=-0.2, max=0.2)
                    p_norm_optim.step()
                    self.p_norm.requires_grad_(False)
                    self.p_norm = self.p_norm.clamp_(min=1.1, max=4.0)
                self.particle_weight.requires_grad_(False)
            with torch.no_grad():
                gdgrad = f_net(self.particle_weight)
                if ep == 0:
                    historical_grad = historical_grad + gdgrad**2
                else:
                    historical_grad = auto_corr * historical_grad + (1 - auto_corr) * gdgrad**2
                adj_grad = (gdgrad)/(fudge_factor + torch.sqrt(historical_grad))
                self.particle_weight = self.particle_weight + self.config.training.master_stepsize * adj_grad
            if ep % 10 == 0 or ep == (self.config.training.n_epoch-1): 
                test_rmse, test_llk = self.model.rmse_llk(self.particle_weight, X_test, y_test, self.y_mean, self.y_std, max_param = 50.0)
                rmse_list.append(test_rmse)
                ll_list.append(test_llk)
                p_norm_list.append(p_norm_item)
                logging.info("iter:[{}]/[{}]  pnorm: {:.4f}, rmse: {:.4f}, llk: {:.4f}".format(ep, self.config.training.n_epoch, p_norm_item, test_rmse, test_llk))
        
        self.model.model_selection(self.particle_weight,X_dev,y_dev,self.y_mean, self.y_std, max_param = 50.0)
        test_rmse, test_llk = self.model.rmse_llk(self.particle_weight, X_test, y_test, self.y_mean, self.y_std, max_param = 50.0)
        rmse_list.append(test_rmse)
        ll_list.append(test_llk)
        p_norm_list.append(p_norm_item)
        logging.info("iter:[{}]/[{}]  pnorm: {:.4f}, after model selection: rmse_list: {:.4f}, ll_list: {:.4f}".format(ep, self.config.training.n_epoch, p_norm_item, test_rmse, test_llk))

        torch.save(np.array(rmse_list), os.path.join(self.config.log, "rmse_list_p={}.pt".format("p_ada" if self.config.optim.p_adaptive else self.p_norm.item())))
        torch.save(np.array(ll_list), os.path.join(self.config.log, "ll_list_p={}.pt".format("p_ada" if self.config.optim.p_adaptive else self.p_norm.item())))
        torch.save(np.array(p_norm_list), os.path.join(self.config.log, "p_norm_list.pt"))


def parse_config():
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config", type=str, default = "smvi/bnn_particle_concrete.yml", help="Path to the config file")
    parser.add_argument('--resume_training',action='store_true',help='Whether to resume training')
    parser.add_argument('--store_path',type=str,default='run',help='Path for saving running related data.')
    parser.add_argument('--permutation_path',type=str,default='run',help='Path of permutation.')
    parser.add_argument('--exp_time',type=int,default=1,)
    parser.add_argument('--master_stepsize',type=float,default=0.0005,)
    parser.add_argument('--f_lr',type=float,default=0.0001,)
    parser.add_argument('--f_iter', type=int,default=10,)
    parser.add_argument('--p_norm_lr',type=float,default=0.0001,)
    parser.add_argument('--p_norm',type=float,default=2.0,)
    parser.add_argument('--p_adaptive',action='store_true',)
    parser.add_argument('--H_alpha',type=float,default= 1.0,)
    parser.add_argument('--beta',type=float,default=0.9,)
    parser.add_argument('--n_epoch',type=int, default=2000,)
    parser.add_argument('--f_optimizer',type=str,default="Adam",)
    parser.add_argument('--f_latent_dim',type=int,default= 300,)
    args = parser.parse_args()
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    new_config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_config.resume_training = args.resume_training
    new_config.store_path = args.store_path
    new_config.exp_time = args.exp_time
    new_config.permutation_path = args.permutation_path
    new_config.training.master_stepsize = args.master_stepsize
    new_config.training.f_iter = args.f_iter
    new_config.training.n_epoch = args.n_epoch
    new_config.model.p_norm = args.p_norm
    new_config.optim.f_lr = args.f_lr
    new_config.optim.p_norm_lr = args.p_norm_lr
    new_config.optim.p_adaptive = args.p_adaptive
    new_config.optim.H_alpha = args.H_alpha
    new_config.optim.f_optimizer = args.f_optimizer
    new_config.model.f_latent_dim = args.f_latent_dim


    return new_config

def main():
    config = parse_config()
    config.log = os.path.join(config.store_path, 'AdaGWG_{}_p_ini_{}_adap_{}_exp_times_{}'.format(config.permutation_path.replace("/","_"), config.model.p_norm, config.optim.p_adaptive, config.exp_time))
    if not config.resume_training:
        os.makedirs(config.log, exist_ok=True)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(config.log, 'stdout.txt'))
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(logging.INFO)
    logging.info(config)
    runner = GWGrunner(config)
    runner.train()

main()