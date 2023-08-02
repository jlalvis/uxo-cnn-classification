import numpy as np
import pandas as pd
import matplotlib.path as mpltPath
import torch
from .net import ConvNet
from .training import probs_to_classes
from .utils import data3dorder, normalize_data, get_coords_data3d, load_sensor_info
#import uxo_utils

def segprob_profile(probs_profile, ncycles, step_size, n_class, nrx):
    nsteps = probs_profile.shape[0]
    L = nsteps*ncycles - (ncycles-step_size)*(nsteps-1)
    probar = np.empty((nsteps,n_class,nrx,L))
    padprob = np.empty((n_class,nrx,L))
    for i in range(nsteps):
        padprob[:] = np.nan
        padprob[:,:,step_size*i:step_size*i+ncycles] = probs_profile[i,:,:,:]
        probar[i,:,:,:] = padprob
    
    sumprobs = np.nansum(probar,axis=0)
    normprobs = sumprobs/np.sum(sumprobs,axis=0)
            
    return probar, normprobs

def seg_profile(labels_profile, ncycles, step_size, nrx):
    nsteps = labels_profile.shape[0]
    L = nsteps*ncycles - (ncycles-step_size)*(nsteps-1)
    archoi = np.empty((nsteps,nrx,L))
    padar = np.empty((nrx,L))
    for i in range(nsteps):
        padar[:] = np.nan
        padar[:,step_size*i:step_size*i+ncycles] = labels_profile[i,:,:]
        archoi[i,:,:] = padar
        
    indnotnan = np.logical_not(np.isnan(archoi[:,0,0]))
        
    maxcat = np.zeros(archoi.shape[1:])
    for i in range(nrx):
        for j in range(L):
            indnotnan = np.logical_not(np.isnan(archoi[:,i,j]))
            maxcat[i,j] = np.bincount(archoi[indnotnan,i,j].astype(int)).argmax()
            
    return maxcat

def classify_data(survey_parameters, class_dict, local_x, local_y, data, times, cnn_pars_file):
    sensor_type = survey_parameters.sensor_type
    #sensorinfo = uxo_utils.load_sensor_info(sensor_type)
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

    use_scaled = False
    use_normalized = True

    ntimes = len(times)
    time_scaling = 1.0*times

    torch.cuda.is_available()
    device = 'cpu'

    in_channels = ntx*3
    layer_geometries3d = [in_channels, 16, 16, 16, 16]
    layer_geometries2d = [128, 128, 128, 128]

    net = ConvNet(layer_geometries3d, layer_geometries2d, ncycles, nrx, ntimes, n_class)

    net.load_state_dict(torch.load(cnn_pars_file))

    net.eval()

    w_step = 1
    
    classified_X = []
    classified_Y = []
    classified_probs = []
    classified_classes = []

    #for line_id in np.unique(lines):
    for line_id in data.files:
        
        X_locs = local_x[line_id]
        Y_locs = local_y[line_id]
        S_data = data[line_id]

        n_windows = (X_locs.shape[1]-ncycles)//w_step + 1
        if n_windows < 1:
            continue

        n = S_data.shape
        net_data = np.zeros((n_windows, n[0], ncycles, n[2]))
        net_y = np.zeros((n_windows, n[0], ncycles))
        net_x = np.zeros((n_windows, n[0], ncycles))

        for i in range(n_windows):
            inds = slice(i*w_step,i*w_step+ncycles)
            net_x[i, :, :] = X_locs[:, inds]
            net_y[i, :, :] = Y_locs[:, inds]
            net_data[i, :, :, :] = S_data[:, inds, :]

        if use_scaled or use_normalized:
            net_data = net_data * time_scaling

        if use_normalized:
            net_data = normalize_data(net_data)

        net_data = data3dorder(net_data, sensor_type)
        net_data_torch = torch.from_numpy(np.float32(net_data))
        x_seg,y_seg = get_coords_data3d(net_x, net_y, net_data.shape[0], ncycles, w_step, sensor_type)

        with torch.no_grad():
            out_field, probs = net(net_data_torch)
            _,segprobs = segprob_profile(probs, ncycles, w_step, n_class, nrx)
            classes = probs_to_classes(probs)
            #maxcat_field = segprob_field.argmax(axis=0) # classify by max sum of probability
            segclasses = seg_profile(classes, ncycles, w_step, nrx) # classify by voting

        classified_X.append(x_seg)
        classified_Y.append(y_seg)
        classified_probs.append(segprobs)
        classified_classes.append(segclasses)
        print(f'line:{line_id}, sliding windows:{n_windows}, classes output shape:{segclasses.shape}')

    return classified_classes, classified_probs, classified_X, classified_Y

def mask_polygon(polygon, lined_x, lined_y, classified_classes, classified_probs):
    n_class = classified_probs[0].shape[0]
    path = mpltPath.Path(polygon)
    bounds_mask = [np.zeros(l.shape,dtype=bool) for l in lined_x]
    for l in range(len(lined_x)): # bool indices for points inside polygon
        points = (np.array((lined_x[l].flatten(),lined_y[l].flatten()))).T
        inside = np.logical_and(path.contains_points(points),(classified_classes[l].flatten() != 0))
        bounds_mask[l] = inside.reshape(lined_x[l].shape)
    # mask x,y and data
    xm = [x[b] for x,b in zip(lined_x,bounds_mask)]
    ym = [y[b] for y,b in zip(lined_y,bounds_mask)]
    data_m = [ld[b] for ld,b in zip(classified_classes,bounds_mask)]
    prob_m = []
    for l,m in enumerate(bounds_mask):
        prob_l = np.zeros((n_class,xm[l].shape[0]))
        for i in range(n_class):
            prob_l[i,:] = classified_probs[l][i][m]
        prob_m.append(prob_l)
    
    return xm, ym, data_m, prob_m


def clean_background(classified_classes, classified_probs, threshold=0.2):
    nlines = len(classified_classes)
    class_not_bg = []
    for i in range(nlines):
        not_bg = np.logical_and(classified_classes[i] != 0, classified_probs[i][0,:,:] < threshold)
        class_not_bg.append((classified_classes[i]*not_bg.astype(int)))
    
    return class_not_bg


def classify_cell(polygon_array, classified_classes, classified_probs, classified_X, classified_Y):
    # For all cells in a 3d-array (cell number x num of vertices x 2)
    n_class = classified_probs[0].shape[0]
    prob_cell = np.zeros((polygon_array.shape[0],n_class))
    x_cell = np.zeros((polygon_array.shape[0]))
    y_cell = np.zeros((polygon_array.shape[0]))
    for j in range(polygon_array.shape[0]):
        cell = polygon_array[j,:,:]
        xm,ym,classes_m,prob_m = mask_polygon(cell, classified_X, classified_Y, classified_classes, classified_probs)
        x_on = [x.mean() for x in xm if x.any()]
        x_cell[j] = sum(x_on)/len(x_on)
        y_on = [y.mean() for y in ym if y.any()]
        y_cell[j] = sum(y_on)/len(y_on)
        # get data from all lines that overlap cell
        indac = [p for p in prob_m if p.any()] 
        fullp = np.hstack(indac)
        prob_cell[j,:] = np.sum(fullp,axis=1)/np.sum(fullp)

    return x_cell, y_cell, prob_cell


def get_diglist(x_cell, y_cell, prob_cell, class_dict, x0, safety_threshold):
    n_class = prob_cell.shape[1]
    results = pd.DataFrame()
    results['Easting'] = x_cell+x0[0]
    results['Northing'] = y_cell+x0[1]
    results[list(class_dict.values())] = prob_cell # add probabilities of each class
    results['obj_class'] = results.iloc[:,2:2+n_class].idxmax(axis=1)
    # replace clutter class with next highest uxo in case clutter probability is below threshold:
    ordprob = prob_cell.argsort(axis=1)
    probmax = prob_cell[np.arange(len(prob_cell)),ordprob[:,-1]]
    clutter_ilist = [i for i in class_dict if 'clutter' in class_dict[i]]
    # get likeliest uxo class after clutter classes:
    clutter_tab = np.zeros((ordprob.shape[0],len(clutter_ilist)), dtype=bool)
    sum_prob_last = prob_cell[np.arange(len(prob_cell)),ordprob[:,-1]]
    clutter_last = np.isin(ordprob[:,-1],clutter_ilist)
    clutter_tab[:,0] = clutter_last & (sum_prob_last<safety_threshold)
    for i in range(len(clutter_ilist)-1):
        sum_prob_last += prob_cell[np.arange(len(prob_cell)),ordprob[:,-i-2]]
        clutter_last = clutter_last & np.isin(ordprob[:,-i-2],clutter_ilist)
        clutter_tab[:,i+1] = clutter_last & (sum_prob_last<safety_threshold)
    clutter_correction = np.copy(ordprob[:,-1])
    sum_clutter_tab = np.sum(clutter_tab, axis=1)
    for i in range(1,len(clutter_ilist)):
        clutter_correction[sum_clutter_tab==i] = ordprob[sum_clutter_tab==i,-i-1]
    # special case when next object is "not TOI":
    clutter_correction[clutter_correction == 0] = ordprob[clutter_correction == 0,-1]
    obj_correction = np.vectorize(class_dict.get)(clutter_correction)
    results['obj_corr'] = obj_correction
    probmax_corr = prob_cell[np.arange(len(prob_cell)),clutter_correction]
    results['probmax_corr'] = probmax_corr
    results['probmax_noc'] = np.where(np.isin(clutter_correction, clutter_ilist),0.0,results['probmax_corr'])
    results.sort_values('probmax_noc',ascending=False,inplace=True)
    num_clutter = sum([1 if 'clutter' in ob else 0 for ob in results['obj_corr']])
    clutter_add = results.iloc[-num_clutter:].sort_values('probmax_corr')
    uxos_add = results.iloc[:-num_clutter]
    results = pd.concat([uxos_add, clutter_add])
    diglist = pd.DataFrame()
    #diglist['Rank'] = np.arange(len(results))+1
    diglist['Easting'] = results['Easting']
    diglist['Northing'] = results['Northing']
    diglist['object'] = results['obj_corr']
    diglist['prob_object'] = results['probmax_corr']
    diglist['dig'] = np.where(results['probmax_noc']>0.0001,1,0)
    diglist = diglist.reset_index(drop=True)

    return results, diglist

