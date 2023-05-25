import numpy as np
import torch
from torch import nn
from torch.nn import functional
import time
#import uxo_utils

from .utils import load_sensor_info
from .net import ConvNet

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(functional.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return (torch.mean(loss) if self.reduction == 'mean'
               else torch.sum(loss) if self.reduction == 'sum'
               else loss)
    
def confusion_matrix(true_labels, predicted_labels):
    n = len(np.unique(true_labels))
    M = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            M[i, j] = np.sum((true_labels == i) & (predicted_labels == j))
    return M

def get_mislabeled(S, labels):
    _, predicted = torch.max(S, dim=1)
    incorrect = (predicted != labels)
    return incorrect.numpy()

def probs_to_classes(probs): 
    _, C = torch.max(probs, axis=1)
    return C

def accuracy(S, labels):
    _, predicted = torch.max(S, dim=1)
    total = np.prod(labels.size())
    correct = (predicted == labels).sum().item()
    return correct/total

def train_net(survey_parameters, class_dict, X_train, C_train, X_test, C_test, times, rseed, nepochs, logfile):
    #sensorinfo = uxo_utils.load_sensor_info(survey_parameters.sensor_type) # if you want to use uxo_utils
    #ntx = len(sensorinfo.transmitters)
    #nrx = len(sensorinfo.receivers)//3 # number of receiver cubes
    sensorinfo = load_sensor_info(survey_parameters.sensor_type)
    ntx = sensorinfo.ntx
    nrx = sensorinfo.nrx # number of receiver cubes
    ymax = survey_parameters.ymax
    y_spacing = survey_parameters.y_spacing
    dy = y_spacing / ntx
    nloc = int(ymax/dy)
    ncycles = int(nloc/ntx)
    n_class = len(class_dict.keys())

    ntimes = len(times)

    if torch.cuda.is_available():
        print("gpu detected, training will use it.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #rseed=9254, 4750, 5293, 9254
    np.random.seed(rseed)
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    torch.cuda.manual_seed_all(rseed)
    torch.backends.cudnn.benchmark = False

    in_channels = ntx*3
    layer_geometries3d = [in_channels, 16, 16, 16, 16] # first part of the net (3d convolutions)
    layer_geometries2d = [128, 128, 128, 128] # second part of the net (2d convolutions)

    net = ConvNet(layer_geometries3d, layer_geometries2d, ncycles, nrx, ntimes, n_class)

    # pass some data through the network (before training):
    with torch.no_grad():
        out, out_probs = net(X_train[:100])

    loss_func = FocalLoss(gamma=2.,reduction='mean')
    #loss_func = nn.CrossEntropyLoss()

    def misfit(C, C_true):
        return loss_func(C, C_true)
        #return ops.sigmoid_focal_loss(C, C_true.float(),gamma=2.,alpha=0.25,reduction='mean')

    n_parout = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # save training log file:
    df = open(logfile, 'w')

    print('Total number of parameters', n_parout)
    print('Total number of parameters', n_parout, file=df)
    print('Total number of training data', C_train.shape[0])
    print('Total number of training data', C_train.shape[0], file=df)

    loss = misfit(out, C_train[:100])
    print('Initial loss = ', loss.detach().numpy())
    print('Initial loss = ', loss.detach().numpy(), file=df)    
    print(f'Check:log({n_class}) = ', np.log(n_class))
    print(f'Check:log({n_class}) = ', np.log(n_class), file=df)

    print(f"\nInitial accuracy: {accuracy(out_probs, C_train[:100])}")
    print(f"\nInitial accuracy: {accuracy(out_probs, C_train[:100])}", file=df)
    print(f"Check random    : {1/n_class}")
    print(f"Check random    : {1/n_class}", file=df)

    import torch.optim as optim
    optimizer = optim.SGD(
        [{'params': net.Kout}, {'params': net.K3d}, {'params': net.K2d}, {'params': net.biasout}], #, {'params': bias}],
        lr = 1e-2, momentum=0
        )

    #running_loss_train = []
    running_loss_test = []
    #running_accuracy_train = []
    running_accuracy_test = []

    # Send model and test and validation datasets to gpu when it is available
    net.to(device)

    X_test = X_test.to(device)
    C_test = C_test.to(device)

    batch_size = 32

    tic = time.perf_counter()
    for epoch in range(nepochs):  # loop over the dataset multiple times

        # zero the parameter gradients
        g = 0.0
        loss = 0.0
        ind = 0
        
        while ind < X_train.shape[0]:  
            optimizer.zero_grad()
            # get the inputs
            inputs = X_train[ind:ind+batch_size, :, :, :, :]
            labels = C_train[ind:ind+batch_size]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward 
            x, _ = net(inputs)
            #print(f'batch: {ind}')
            lossi = misfit(x, labels)
            if ind==0:
                loss = lossi
            else:
                loss += lossi
            lossi.backward()
            optimizer.step()
            ind += batch_size
            
        with torch.no_grad():
            xtest, probs_test = net(X_test)
            loss_test = misfit(xtest, C_test)

            #xtrain, probs_train = net(X_train)
            #loss_train = misfit(xtrain, C_train)

            #accuracy_train = accuracy(probs_train, C_train)
            accuracy_test = accuracy(probs_test, C_test)

            #running_loss_train.append(loss_train)
            running_loss_test.append(loss_test)
            #running_accuracy_train.append(accuracy_train)
            running_accuracy_test.append(accuracy_test)
        
            print(f'epoch: {epoch:3d}, loss_test: {loss_test:2.3e}, accuracy_test: {accuracy_test:0.3f}') # {loss_train:2.3e} {accuracy_train:0.3f}
            print(f'epoch: {epoch:3d}, loss_test: {loss_test:2.3e}, accuracy_test: {accuracy_test:0.3f}', file=df)

    toc = time.perf_counter()
    print(f'Finished Training in {toc-tic:0.4f} seconds')
    print(f'Finished Training in {toc-tic:0.4f} seconds', file=df)
    df.close()

    return net
