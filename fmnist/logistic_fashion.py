#!/usr/bin/python


import argparse
import numpy as np
import random
## import pytorch modules
import torch

import model

def main():
    parser = argparse.ArgumentParser(description='FashionMNIST federated HMC/LD')
    
    # training hyperparameters
    parser.add_argument('-total_step', default=1000, type=int, help='totoal number of HMC / LD steps')
    parser.add_argument('-T', default=0.05, type=float, help='Temperature for sampling')
    parser.add_argument('-batch_train', default=2000, type=int, help='Batch size for training')
    parser.add_argument('-batch_test', default=2000, type=int, help='Batch size for testing')
    parser.add_argument('-lr', default=1e-7, type=float, help='sampling step size')
    
    # saving hyperparameters
    parser.add_argument('-save_size', default=100, type=int, help='number of samples to save in one file')
    parser.add_argument('-save_name', type=str, help='name of the file to save predicted proabilities to')
    parser.add_argument('-save_gap', default=10, type=int, help='the gap between two saved samples')
    parser.add_argument('-save_test_name', type=str, help='name of the files to save the output test results')
    
    # evaluation hyperparameters
    parser.add_argument('-bin_size', default=20, type=int, help='number of bins to calculation calibration')
    
    # HMC/LD hyperparameters
    #parser.add_argument('-LD', default=1, type=int, help='whether LD or not. 1 corresponds to LD, o.w. HMC')
    parser.add_argument('-leapfrog_step', default=1, type=int, help='number of leapfrog steps for HMC')
    
    # Federated learning hyperparameters
    parser.add_argument('-federated', default=1, type=int, help='whether federated or not')
    parser.add_argument('-num_client', default=1, type=int, help='number of clients')
    parser.add_argument('-local_step', default=1, type=int, help='number of local steps')
    parser.add_argument('-adaptive', default=0, type=int, help='1 for adaptive AFHMC, 0 for standard FA-HMC')
    parser.add_argument('-fast', default=0, type=int, help='1 = preload-to-GPU fast trainer (no DataLoader)')
    parser.add_argument('-split', default='iid', choices=['iid', 'dirichlet'],
                        help='per-client partition (fast path only): iid random shuffle, or Dirichlet over labels')
    parser.add_argument('-dirichlet_alpha', default=0.1, type=float,
                        help='Dirichlet concentration; smaller = more label-skewed (0.1 = SCAFFOLD canonical)')

    # Other hyperparameters
    parser.add_argument('-data', default='FashionMNIST', dest='data', help='FashionMNIST')
    parser.add_argument('-seed', default=random.randint(1, 1e6), type=int, help='Random Seed')
    parser.add_argument('-gpu', default=0, type=int, help='Default GPU')


    pars = parser.parse_args()
    """ Step 0: Numpy printing setup and set GPU and Seeds """
    print(pars)
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    try:
        torch.cuda.set_device(pars.gpu)
    except: # in case the device has only one GPU
        torch.cuda.set_device(0) 
    torch.manual_seed(pars.seed)
    torch.cuda.manual_seed(pars.seed)
    np.random.seed(pars.seed)
    random.seed(pars.seed)
    torch.backends.cudnn.deterministic = True

    """ Step 1: Preprocessing """
    if not torch.cuda.is_available():
        exit("CUDA does not exist!!!")
    net = model.MLP().cuda()
    net_server = model.MLP().cuda()

    nets = []
    for _ in range(pars.num_client):
        client_net = model.MLP().cuda()
        client_net.load_state_dict(net.state_dict())
        nets.append(client_net)
    
    """ Step 2: Load Data and Train """
    if pars.federated != 1 or pars.fast != 1:
        raise ValueError('This export keeps only the federated fast path; pass -federated 1 -fast 1.')
    from tools_fast import loader_federated_fast, validation_fast
    from trainer_fast import training_federated_fast
    train_x, train_y, clients, test_x, test_y, train_size = loader_federated_fast(pars)
    test_results_initial, _ = validation_fast(net, test_x, test_y, pars.bin_size, batch_test=pars.batch_test)
    test_ave, test_trace = training_federated_fast(
        nets, net_server, train_x, train_y, clients, test_x, test_y, train_size, pars
    )
    
    test_results_initial = np.array(test_results_initial)
    np.savez(pars.save_test_name, seed=pars.seed, test_initial=test_results_initial, test_val=np.array(test_ave[0:5]), entropies=test_ave[5], bin_conf=test_ave[6], bin_accu=test_ave[7], test_trace=test_trace)
    torch.save(net_server.cpu().state_dict(), pars.save_name+'model.pt')
    

if __name__ == "__main__":
    main()
