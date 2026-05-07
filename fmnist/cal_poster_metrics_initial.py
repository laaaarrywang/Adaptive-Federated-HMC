#!/usr/bin/python


import argparse
import numpy as np
import random
## import pytorch modules
import torch
import torch.nn.functional as Func

from tools import loader
#from trainer import training, training_federated

def evaluation(probs_all, test_loader, M):
    nll_all = 0
    accu = 0
    Brier = 0
    test_size = len(test_loader.dataset)
    entropies = np.zeros(test_size)
    conf_accu = np.zeros((test_size, 2))
    probs_all = probs_all.cuda()
    output_all = probs_all.log()
    with torch.no_grad():
        count = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            
            batch_size = len(labels)
            output = output_all[count:(count+batch_size), ]
            probs = probs_all[count:(count+batch_size), ]
            nll_all = nll_all + Func.nll_loss(output, labels, reduction='sum').data.item()
            Brier = Brier + torch.sum((probs.data - Func.one_hot(labels.data, num_classes=output.shape[1]))**2).item()
            max_output = probs.data.max(1)
            conf_accu[count:(count+batch_size), 0] = max_output[0].cpu().numpy()
            pred = max_output[1]
            matched = pred.eq(labels.data)
            accu = accu + matched.sum().item()
            conf_accu[count:(count+batch_size), 1] = matched.cpu().numpy()
            entropies[count:(count+batch_size)] = - torch.sum(output * probs, axis=1).data.cpu().numpy()
            count = count + batch_size
            
    accu = accu / test_size
    Brier = Brier / test_size
    entropy = np.mean(entropies).item()
    
    bins = np.linspace(0, 1, M+1)
    conf_accu[:,0]
    bin_assign = np.digitize(conf_accu[:,0], bins)
    bin_conf = np.zeros(M)
    bin_accu = np.zeros(M)
    bin_size = np.zeros(M, dtype=int)
    for j in range(1, M+1):
        bin_size[j-1] = np.sum(bin_assign==j)
        if bin_size[j-1] > 0:    
            bin_conf[j-1] = np.mean(conf_accu[bin_assign==j, 0])
            bin_accu[j-1] = np.mean(conf_accu[bin_assign==j, 1])
    
    ECE = np.sum(bin_size * np.abs(bin_conf - bin_accu)) / test_size
    MCE = np.max(np.abs(bin_conf - bin_accu))
    
    print("Accuracy: ", accu)
    print("Brier score: ", Brier)
    print("NLL: ", nll_all)
    print("ECE: ", ECE)
    print("MCE: ", MCE)
    
    return nll_all, accu, Brier, ECE, MCE, entropy


def main():
    parser = argparse.ArgumentParser(description='CIFAR10 federated HMC/LD')
    
    # training hyperparameters
    parser.add_argument('-batch_train', default=2000, type=int, help='Batch size for training')
    parser.add_argument('-batch_test', default=2000, type=int, help='Batch size for testing')
    
    parser.add_argument('-local_step', default=1, type=int, help='number of local steps')
    parser.add_argument('-leapfrog_step', default=1, type=int, help='number of leapfrog steps for HMC')
    
    # saving hyperparameters
    parser.add_argument('-save_size', default=100, type=int, help='number of samples to save in one file')
    parser.add_argument('-save_name', type=str, help='name of the file to save predicted proabilities to')
    parser.add_argument('-cal_gap', default=1, type=int, help='the gap between two saved samples')
    #parser.add_argument('-cal_number', default=100, type=int, help='the total number of parameters to calculate')
    parser.add_argument('-save_test_name', type=str, help='name of the files to save the output test results')
    
    parser.add_argument('-save_gap', default=10, type=int, help='the gap between two saved samples')
    
    parser.add_argument('-start_id', default=1, type=int, help='the starting file index')
    parser.add_argument('-end_id', default=1, type=int, help='the ending file index')
    
    # evaluation hyperparameters
    parser.add_argument('-bin_size', default=20, type=int, help='number of bins to calculation calibration')
    
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
    
    
    """ Step 2: Load Data and Train """
    train_loader, test_loader, train_size = loader(pars.batch_train, pars.batch_test, pars)
    
    cal_number = pars.save_size // pars.cal_gap * (pars.end_id - pars.start_id + 1)
    
    test_single = np.zeros((cal_number, 6))
    test_poster = np.zeros((cal_number, 6))
    
    output_gap = np.lcm(pars.local_step, pars.leapfrog_step) * pars.save_gap
    comm_gap = np.lcm(pars.local_step, pars.leapfrog_step) // pars.local_step * pars.save_gap
    comm_round = np.arange(1, cal_number + 1) * comm_gap * pars.cal_gap
    numerical_iter = np.arange(1, cal_number + 1) * output_gap * pars.cal_gap
    
    probs_sum = 0
    count = 0
    for i in range(pars.start_id, pars.end_id + 1):
        ptfile = torch.load(pars.save_name+str(i)+'.pt')
        id_used = np.arange(pars.cal_gap, pars.save_size + 1, pars.cal_gap) - 1
        for j in range(len(id_used)):
            count = count + 1
            probs_new = ptfile[id_used[j]]
            probs_sum = probs_sum + probs_new
            test_single[count-1,] = evaluation(probs_new, test_loader, pars.bin_size)
            test_poster[count-1,] = evaluation(probs_sum / count, test_loader, pars.bin_size)
    
    test_initial = np.load(pars.save_name+'test.npz')['test_initial']
    test_single = np.vstack((test_initial, test_single))
    test_poster = np.vstack((test_initial, test_poster))
    comm_round = np.insert(comm_round, 0, 0)
    
    np.savez(pars.save_test_name, seed=pars.seed, test_single=test_single, test_poster=test_poster, comm_round=comm_round, numerical_iter=numerical_iter)
    

if __name__ == "__main__":
    main()
