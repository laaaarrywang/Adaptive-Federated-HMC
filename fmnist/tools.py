import numpy as np

## import pytorch modules
import torch
import torch.nn.functional as Func
import torch.nn as nn
         
import torch.utils.data as data
import torchvision.datasets as datasets
 
import transforms

from torch.utils.data import random_split


def loader_federated(train_size, test_size, args):
    if args.data.startswith('Fashion'):
        dataloader = datasets.FashionMNIST
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        #transform_train = transforms.Compose([
        #    transforms.RandomCrop(32, padding=4),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #    transforms.RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.3, ),
        #])
        
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        #transform_test = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #])
    else:
        exit('Unknown dataset')

    trainset = dataloader('./data/' + args.data, train=True, download=True, transform=transform_train)
    train_length = len(trainset)
    client_length = int(np.floor(train_length / args.num_client))
    split_size = [client_length] * (args.num_client - 1)
    split_size.append(train_length - sum(split_size))
    clients_set = random_split(trainset, split_size, generator=torch.Generator().manual_seed(args.seed))
    clients_loader = []
    for i in range(args.num_client):
        clients_loader.append(data.DataLoader(clients_set[i], batch_size=train_size, shuffle=True, num_workers=0))
    #train_loader = data.DataLoader(trainset, batch_size=train_size, shuffle=True, num_workers=0) # num_workers=0 is crucial for seed
    """ caution: no shuffle on test dataset """
    testset = dataloader(root='./data/' + args.data, train=False, download=True, transform=transform_test)
    #testset = dataloader(root='./data/' + 'MNIST', train=False, download=True, transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=test_size, shuffle=False, num_workers=0)
    return clients_loader, test_loader, split_size, train_length


def loader(train_size, test_size, args):
    if args.data.startswith('Fashion'):
        dataloader = datasets.FashionMNIST
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        #transform_train = transforms.Compose([
        #    transforms.RandomCrop(32, padding=4),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #    transforms.RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.3, ),
        #])
        
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        #transform_test = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #])
    else:
        exit('Unknown dataset')

    trainset = dataloader('./data/' + args.data, train=True, download=True, transform=transform_train)
    train_length = len(trainset)
    train_loader = data.DataLoader(trainset, batch_size=train_size, shuffle=True, num_workers=0) # num_workers=0 is crucial for seed
    """ caution: no shuffle on test dataset """
    testset = dataloader(root='./data/' + args.data, train=False, download=True, transform=transform_test)
    #testset = dataloader(root='./data/' + 'MNIST', train=False, download=True, transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=test_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, train_length


def evaluation_net(net, test_loader, M):
    nll_all = 0
    accu = 0
    Brier = 0
    test_size = len(test_loader.dataset)
    entropies = np.zeros(test_size)
    conf_accu = np.zeros((test_size, 2))
    with torch.no_grad():
        count = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            
            batch_size = len(labels)
            output = Func.log_softmax(net(images))
            probs = output.exp()
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
    
    return nll_all, accu, Brier, ECE, MCE, entropies, bin_conf, bin_accu


def evaluation(probs_all, test_loader, M):
    nll_all = 0
    accu = 0
    Brier = 0
    test_size = len(test_loader.dataset)
    entropies = np.zeros(test_size)
    conf_accu = np.zeros((test_size, 2))
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
    
    return nll_all, accu, Brier, ECE, MCE, entropies, bin_conf, bin_accu
    

def validation(net, test_loader, M):
    nll_all = 0
    accu = 0
    Brier = 0
    test_size = len(test_loader.dataset)
    entropies_sum = 0
    conf_accu = np.zeros((test_size, 2))
    probs_all = torch.zeros(test_size, 10).cuda()
    count = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            
            batch_size = len(labels)
            output = Func.log_softmax(net(images))
            probs = output.exp()
            probs_all[count:(count+batch_size), ] = probs.detach().data
            nll_all = nll_all + Func.nll_loss(output, labels, reduction='sum').data.item()
            Brier = Brier + torch.sum((probs.data - Func.one_hot(labels.data, num_classes=output.shape[1]))**2).item()
            max_output = probs.data.max(1)
            conf_accu[count:(count+batch_size), 0] = max_output[0].cpu().numpy()
            pred = max_output[1]
            matched = pred.eq(labels.data)
            accu = accu + matched.sum().item()
            conf_accu[count:(count+batch_size), 1] = matched.cpu().numpy()
            entropies_sum = entropies_sum - torch.sum(torch.sum(output * probs, axis=1).detach().data).item()
            count = count + batch_size
            
    accu = accu / test_size
    Brier = Brier / test_size
    entropy = entropies_sum / test_size
    
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
    print("Entropy: ", entropy)
    
    return (nll_all, accu, Brier, ECE, MCE, entropy), probs_all
    
    
    
