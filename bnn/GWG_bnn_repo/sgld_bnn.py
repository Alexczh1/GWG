import os
import numpy as np
import torch
import argparse
import logging
from models.bnn_models import BNN

class SGLD_BNN(object):
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.permutation_path = config.permutation_path
        self.loop = config.loop
        self.epsilon_0 = config.epsilon_0
        self.alpha = config.alpha
        self.dataset = config.data_name
        self.data_batch_size = config.data_batch_size
        self.latent_dim = 50
        self.num_particle = 100
    def init_weights(self, X_train, y_train):
        w1 = 1.0 / np.sqrt(X_train.shape[1] + 1) * torch.randn(self.num_particle, X_train.shape[1] * self.latent_dim).to(self.device)
        b1 = torch.zeros((self.num_particle, self.latent_dim)).to(self.device)
        w2 = 1.0 / np.sqrt(self.latent_dim + 1) * torch.randn(self.num_particle, self.latent_dim).to(self.device)
        b2 = torch.zeros((self.num_particle,1)).to(self.device)
        loglambda = torch.ones((self.num_particle,1)).to(self.device) * np.log(np.random.gamma(1, 0.1))
        loggamma = torch.ones((self.num_particle,1)).to(self.device)
        for i in range(self.num_particle):
            with torch.no_grad():
                ridx = np.random.choice(range(X_train.shape[0]), np.min([X_train.shape[0], 1000]), replace = False)
                y_hat = self.model.predict_y(torch.cat([w1[i][None,:],b1[i][None,:],w2[i][None,:],b2[i][None,:]],dim=1).to(self.device), X_train[ridx,:], self.y_mean, self.y_std, scale = False)
                loggamma[i] = -torch.log(torch.mean((y_hat - y_train[ridx,:])**2,1))
        return torch.cat([w1,b1,w2,b2, loggamma, loglambda],dim=1).to(self.device)
    def get_datasets(self, dataset, test_size=0.1):
        if dataset == 'boston':
            data = np.loadtxt('data/boston_housing.txt')
        X = torch.from_numpy(data[:, :-1]).float().to(self.device)

        y = torch.from_numpy(data[:, -1]).float().to(self.device)
        y = y.unsqueeze(1)

        X = X.to(self.config.device)
        y = y.to(self.config.device)

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
    def train(self):
        rmse_list = []
        X_train, y_train, X_dev, y_dev, X_test, y_test = self.get_datasets(self.dataset)
        ''' load dataset '''
        self.model = BNN("cpu", X_train.shape[1])
        self.scale_sto = X_train.shape[0]/self.data_batch_size
        self.particle_weight = self.init_weights(X_train, y_train)
        size_train = X_train.shape[0]
        for t in range(self.loop):
            batch_index = [ i % size_train for i in range(t*self.data_batch_size, (t + 1)*self.data_batch_size)]
            x = X_train[batch_index]
            y = y_train[batch_index]
            compu_targetscore = self.model.score(self.particle_weight, x, y, self.scale_sto)
        
            learn_rate = np.max((self.epsilon_0 /(t+1)**self.alpha, 1e-8))   
            self.particle_weight = self.particle_weight + learn_rate/2 * compu_targetscore + np.sqrt(learn_rate) * torch.randn([self.particle_weight.shape[0],self.particle_weight.shape[1]]).to(self.device)
            if (t+1)%100==0 or t == 0:
                svgd_rmse, svgd_ll = self.model.rmse_llk(self.particle_weight, X_test, y_test, self.y_mean, self.y_std, max_param = 50.0)
                logging.info("t: {}, rmse: {:.4f}, test_llk:{:.4f}".format(t, svgd_rmse, svgd_ll))
                rmse_list.append(np.array([t,svgd_rmse]))
        t = t + 1
        self.model.model_selection(self.particle_weight,X_dev,y_dev,self.y_mean, self.y_std, max_param = 50.0)
        svgd_rmse, svgd_ll = self.model.rmse_llk(self.particle_weight, X_test, y_test, self.y_mean, self.y_std, max_param = 50.0)
        rmse_list.append(np.array([t,svgd_rmse]))
        logging.info("t: {}, after model selection: rmse: {:.4f}, test_llk:{:.4f}".format(t, svgd_rmse, svgd_ll))
        torch.save(np.array(rmse_list), os.path.join(self.config.log, 'rmse_list.pt'))



def main():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--store_path',
                        type=str,
                        default='run',
                        help='Path for saving running related data.')
    parser.add_argument('--permutation_path',
                        type=str,
                        default='run',
                        help='Path of permutation.')
    parser.add_argument('--epsilon_0',
                        type=float,
                        default=0.0005)
    parser.add_argument('--alpha',
                        type=float,
                        default=0)
    parser.add_argument('--data_batch_size',
                        type=int,
                        default=100)
    parser.add_argument('--data_name',
                        type=str,
                        default="boston")
    parser.add_argument('--loop',
                        type=int,
                        default=1000)
    parser.add_argument('--exp_time',
                        type=int,
                        default=1)                 
    config = parser.parse_args()
    config.device = "cpu"
    config.store_path = "run_{}".format(config.data_name)
    config.log = os.path.join(config.store_path, 'sgld_logs_{}_time_{}'.format(config.permutation_path.replace("/","_"),config.exp_time))
    os.makedirs(config.log, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    logging.info("config:{}".format(config))
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
    runner = SGLD_BNN(config)
    runner.train()

main()