# uxo-cnn-classification

This repository has the implementation of a fully convolutional neural network (CNN) to classify unexploded ordnance (UXO) from multi-channel electromagnetic induction (EMI) data.

The training datasets are synthetic (generated with a forward model for time domain EMI) and target labels should be given in a segmentation-like architecture.

The general workflow is as follows:

1. Train the binary CNN with synthetic data using either `training-script-binary.py` or `training-binary-with-synthetic-data.ipynb`. The only difference between these two files is that the notebook generates some plots.

2. Obtain binary classification maps (metallic object vs background) and manually create cells around each target identified.

3. Crop out data within the cells and from the remaining data estimate background noise (which is usually spatially correlated).

4. Train the multi-class classifier with synthetic data + background noise using either `training-script.py` or `training-with-synthetic-data.ipynb`. The only difference between these two files is that the notebook generates some plots.

5. Obtain multi-class classification maps showing UXO type (40, 60, 81, 105 or 155 mm) or clutter. Additionally, obtain a dig list based on probability values from the CNN.

This workflow has been applied to two field cases (2021 and 2022) with a calibration grid and a blind grid each (*train_case* refers to these different scenarios).

Two different background removal methods have been applied (*bg_case* refers to these two methods).

### Input files for training:
`labmask*.npy` target labels in segmentation-like structure  
`data*.npy` data files containing time domain EMI data (in the case of the multiclass classification these files should already include the correlated background noise)  
`times.npy` time value for each time channel for EMI data

### Output files from training:
`trainlog*` log file containing loss history  
`net*.pth` parameters estimated for the CNN

### Input files for classification:
`net*.pth` previously trained CNN  
`masked_data*.npz` field data, pre-processed so that data is arranged according to transmitter firing sequence  
`masked_x*.npz` central x coordinate assigned to a full transmitter cycle  
`masked_y*.npz` central y coordinate assigned to a full transmitter cycle

### Output files from classification:
`classcells*.png` classification map (binary or multiclass)  
`diglist*.xlsx` spreadsheet file with UXO dig list

