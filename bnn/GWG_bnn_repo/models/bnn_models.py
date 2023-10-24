import numpy as np
import torch


class BNN(object):
    name = "Bnn"
    def __init__(self, device, d, a = 1, b = 0.1, n_hidden = 50):
        self.device = device
        self.a = a
        self.b = b
        self.n_hidden = n_hidden
        self.d = d
        self.dim_vars = (self.d + 1) * self.n_hidden + (self.n_hidden + 1) + 2
        self.dim_wb = self.dim_vars - 2
    def logp(self, Z, batchdataset, batchlabel, scale_sto = 1, max_param = 50.0):
        """
        return the log posterior distribution P(W, log_gamma, log_lambda|Y,X)
        """
        log_gamma = Z[:,-2]
        log_lambda = Z[:,-1]
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        lambda_ = torch.exp(log_lambda).clamp(max=max_param)
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-3].reshape(-1,1) # [B, 1]
        dnn_predict = (torch.matmul(torch.max(torch.matmul(batchdataset, W1) + b1[:,None,:], torch.tensor([0.0]).to(self.device)), W2) + b2[:,None,:])   # [B, n, 1]
        log_lik_data = -0.5 * batchdataset.shape[0] * (np.log(2*np.pi) - log_gamma) - (gamma_/2) * torch.sum(((dnn_predict-batchlabel).squeeze(2))**2, 1)
        log_prior_data = (self.a - 1) * log_gamma - self.b * gamma_ + log_gamma
        log_prior_w = -0.5 * self.dim_wb * (np.log(2*np.pi) - log_lambda) - (lambda_/2)*((W1**2).sum((1,2)) + (W2**2).sum((1,2)) + (b1**2).sum(1) + (b2**2).sum(1)) \
                        + (self.a-1) * log_lambda - self.b * lambda_ + log_lambda
        return (log_lik_data * scale_sto + log_prior_data + log_prior_w)
        
    def score(self, Z, batchdataset, batchlabel, scale_sto = 1, max_param = 50.0):
        batch_Z = Z.shape[0]
        num_data = batchdataset.shape[0]

        log_gamma = Z[:,-2].reshape(-1,1) # [B, 1]
        log_lambda = Z[:,-1].reshape(-1,1)
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        lambda_ = torch.exp(log_lambda).clamp(max=max_param)
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-3].reshape(-1,1) # [B, 1]

        dnn_onelinear = torch.matmul(batchdataset, W1) + b1[:,None,:]
        dnn_relu_onelinear = torch.max(dnn_onelinear, torch.tensor([0.0]).to(self.device))
        dnn_grad_relu = (torch.sign(dnn_onelinear) + 1)/2 # shape = [B, n, hidden]
        dnn_predict = (torch.matmul(dnn_relu_onelinear, W2) + b2[:,None,:]) # shape = [B,n,1]
        
        
        nabla_predict_b1 = dnn_grad_relu * W2.transpose(1,2) # [B, n, hidden]
        nabla_predict_W1 = nabla_predict_b1[:,:,None,:] * batchdataset[None,:,:,None] # [B,n,d, hidden] 
        nabla_predict_W2 = dnn_relu_onelinear # [B,n, hidden]
        nabla_predict_b2 = torch.ones_like(dnn_predict).to(self.device) # [B,n,1]

        nabla_predict_wb = torch.cat((nabla_predict_W1.reshape(batch_Z, num_data, -1), nabla_predict_b1, nabla_predict_W2, nabla_predict_b2),dim=2)
        nabla_wb = scale_sto * gamma_ * ((batchlabel - dnn_predict) * nabla_predict_wb).sum(1) - lambda_ * Z[:,:-2]
        nabla_log_gamma = scale_sto * (0.5 * num_data - (gamma_/2) * torch.sum((dnn_predict-batchlabel)**2, 1)) + (self.a - 1) - self.b * gamma_ + 1 #[B, 1]
        nabla_log_lambda = 0.5 * self.dim_wb - lambda_/2 * (Z[:,:-2]**2).sum(1).unsqueeze(1) + (self.a - 1) - self.b * lambda_ + 1 # [B,1]
        return torch.cat((nabla_wb, nabla_log_gamma, nabla_log_lambda), dim=1)      # shape = [B, self.dim_vars]
    def rmse_llk(self, Z, batchdataset, batchlabel, mean_y_train, std_y_train, max_param = 50.0):
        log_gamma = Z[:,-2].reshape(-1,1) # [B, 1]
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-3].reshape(-1,1) # [B, 1]
        dnn_predict = (torch.matmul(torch.max(torch.matmul(batchdataset, W1) + b1[:,None,:], torch.tensor([0.0]).to(self.device)), W2) + b2[:,None,:])
        dnn_predict_true = dnn_predict * std_y_train + mean_y_train # [B, n, 1]
        predict_mean = dnn_predict_true.mean(0)
        test_rmse = torch.sqrt(((predict_mean - batchlabel)**2).mean())
        logpy_xz = -0.5 * (np.log(2*np.pi) - log_gamma[:,None,:]) - 0.5 * gamma_[:, None, :] * (dnn_predict_true - batchlabel[None, :, :])**2
        test_llk = (torch.logsumexp(logpy_xz.squeeze(2), dim=0).mean() - np.log(Z.shape[0]))
        return test_rmse.item(), test_llk.item()
    def predict_y(self, Z, batchdataset, mean_y_train, std_y_train, max_param = 50.0, scale = True):
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-3].reshape(-1,1) # [B, 1]
        dnn_predict = (torch.matmul(torch.max(torch.matmul(batchdataset, W1) + b1[:,None,:], torch.tensor([0.0]).to(self.device)), W2) + b2[:,None,:])
        if scale:
            dnn_predict_true = dnn_predict * std_y_train + mean_y_train
            return dnn_predict_true
        else:
            return dnn_predict
    def model_selection(self, Z, batchdataset, batchlabel, mean_y_train, std_y_train, max_param = 50.0):
        log_gamma = Z[:,-2].reshape(-1,1) # [B, 1]
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-3].reshape(-1,1) # [B, 1]
        dnn_predict = (torch.matmul(torch.max(torch.matmul(batchdataset, W1) + b1[:,None,:], torch.tensor([0.0]).to(self.device)), W2) + b2[:,None,:])
        dnn_predict_true = dnn_predict * std_y_train + mean_y_train # [B, n, 1]
        log_gamma_heu = -torch.log(((dnn_predict_true - batchlabel[None, :, :])**2).mean(1))
        for i in range(dnn_predict_true.shape[0]):
            def f_log_lik(loggamma): return torch.sum(torch.log(torch.sqrt(torch.exp(loggamma)) * torch.exp( -1/2 * (dnn_predict_true[i]- batchlabel)**2 * torch.exp(loggamma) )) )
            lik1 = f_log_lik(log_gamma[i])
            lik2 = f_log_lik(log_gamma_heu[i])
            if lik2 > lik1:
                Z[i:,-2] = log_gamma_heu[i]
        return Z
