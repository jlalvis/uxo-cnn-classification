import numpy as np
import os
import torch
import time
from torch import nn
from torch.nn import functional
import random

import matplotlib.pyplot as plt
from matplotlib import cm as cmap
from matplotlib import colors
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d

#import uxo_utils

from uxo_cnn_classification import (
    train_net, SurveyParameters, 
    normalize_data, data3dorder,
    accuracy, get_mislabeled,
    confusion_matrix, load_sensor_info
)

# initialization random seed for training and number of epochs:
seeds = [4750, 5293, 9254]
rseed = seeds[0]
nepochs = 10

# sequim group 2022
class_dict = {
    0: "not TOI",
    1: "UXO",
}
survpars = SurveyParameters(
    sensor_type = 'ultratema',
    ymax = 4.5,
    y_spacing = 0.3,
    min_standoff = 0.4,
    #noise_file = ''
)
train_case = 'sequimgroup2022'
bg_case = 'a'

# # sequim group 2021
# class_dict = {
# class_dict = {
#     0: "not TOI",
#     1: "UXO",
# }
# survpars = SurveyParameters(
#     sensor_type = 'ultratema',
#     ymax = 4.5,
#     y_spacing = 0.3,
#     min_standoff = 0.5,
#     #noise_file = ''
# )
# train_case = 'sequimgroup2021'
# bg_case = 'a'

sensorinfo = load_sensor_info(survpars.sensor_type)
n_class = len(class_dict.keys())
#ntx = len(sensorinfo.transmitters)
#nrx = len(sensorinfo.receivers)//3 # number of receiver cubes
ntx = sensorinfo.ntx
nrx = sensorinfo.nrx
dy = survpars.y_spacing / ntx
nloc = int(survpars.ymax/dy)
ncycles = int(nloc/ntx)

data_directory = f"/media/jorge/T7/databin_{train_case}{bg_case}_s100"

# training dataset is split in different files (mainly to handle memory more efficiently while simulating synthetic data)
nfiles = 2 # here, the last file will be used as test data

#pos = np.load(os.path.sep.join([data_directory, "pos.npy"]))
times = np.load(os.path.sep.join([data_directory, "times.npy"]))
#sensor_table = np.load(os.path.sep.join([data_directory, "sensor_table.npy"]))

labstrain_list = []
for i in range(nfiles-1): # number of files
    labstrain_list.append(np.load(os.path.sep.join([data_directory, f"labels{i}.npy"])))
    
print('loading training data...')

labels_train = np.vstack(labstrain_list)
labels_test = np.load(os.path.sep.join([data_directory, f"labels{nfiles-1}.npy"]))

ntimes = len(times)

labmask_list = []
for i in range(nfiles-1): # number of files
    labmask_list.append(np.load(os.path.sep.join([data_directory, f"labmask{i}.npy"])))
    
labmask_train = np.vstack(labmask_list)

labmask_test = np.load(os.path.sep.join([data_directory, f"labmask{nfiles-1}.npy"]))

dtrain_list = []
for i in range(nfiles-1): # number of files
    dtrain_list.append(np.load(os.path.sep.join([data_directory, f"data{i}.npy"])))
    
data_train = np.vstack(dtrain_list)
del dtrain_list # to free some memory
data_test = np.load(os.path.sep.join([data_directory, f"data{nfiles-1}.npy"]))

use_scaled = False
use_normalized = True

ntimes = len(times)

# transform labels and masks to 0 and 1:
labels_train = (labels_train>0).astype(int)
labels_test = (labels_test>0).astype(int)
labmask_train = (labmask_train>0).astype(int)
labmask_test = (labmask_test>0).astype(int)

print('normalizing training data...')

if use_scaled or use_normalized:
    time_scaling = (times)
    scaled_data_train = data_train * time_scaling
    scaled_data_test = data_test * time_scaling
    del data_train, data_test

if use_normalized:
    normalized_data_train = normalize_data(scaled_data_train)
    del scaled_data_train
    normalized_data_test = normalize_data(scaled_data_test)
    del scaled_data_test

normalized_data_train = data3dorder(normalized_data_train, survpars.sensor_type)
normalized_data_test = data3dorder(normalized_data_test, survpars.sensor_type)

if use_scaled is True: 
    X_train = torch.from_numpy(np.float32(scaled_data_train))
    X_test = torch.from_numpy(np.float32(scaled_data_test))

elif use_normalized is True: 
    X_train = torch.from_numpy(np.float32(normalized_data_train))
    X_test = torch.from_numpy(np.float32(normalized_data_test))

else: 
    X_train = torch.from_numpy(np.float32(data_train))
    X_test = torch.from_numpy(np.float32(data_test))

C_train = torch.from_numpy(np.float32(labmask_train)).long()
C_test = torch.from_numpy(np.float32(labmask_test)).long()

if use_normalized:
    del normalized_data_train, normalized_data_test

cnn_pars_dir = os.path.join(os.getcwd(), 'cnn_parameters')

if not os.path.exists(cnn_pars_dir):
    os.mkdir(cnn_pars_dir)

logfile = os.path.join(cnn_pars_dir, f'trainlogbin_{train_case}{bg_case}_{rseed}_s100_script')
net = train_net(survpars, class_dict, X_train, C_train, X_test[:1000], C_test[:1000], times, rseed, nepochs, logfile)

# save trained parameters:
torch.save(net.state_dict(), os.path.join(cnn_pars_dir, f'netbin{train_case}{bg_case}_{rseed}_s100_script.pth'))