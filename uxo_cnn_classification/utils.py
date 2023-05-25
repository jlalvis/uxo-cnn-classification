import numpy as np
from dataclasses import dataclass
#import uxo_utils

@dataclass
class OrdnanceInfo:
    ordnance_library: str
    class_dict: dict # including clutter objects
    clutter_list: list # list of clutter groups (each group has one or more types: 'Sphere','Rod','Disk')
    clutter_polarizations: dict # axial and transversal for each type of object


@dataclass
class SurveyParameters:
    sensor_type: str
    ymax: float # length of along-line window we collect data in
    y_spacing: float # spacing between each transmitter firing
    min_standoff: float
    noise_file: str = '' # only needed for synthetic data simulation


@dataclass
class ParameterRanges:
    depth_range: np.array
    yaw_range: np.array
    pitch_range: np.array
    roll_range: np.array
    noise_amplitudes: np.array
    x_range: np.array = np.r_[0, 0] # from system footprint
    y_range: np.array = np.r_[0, 0] # from system footprint


@dataclass
class Sensor: # replacement class for "SensorInfo", if BTInvert is available one can use "uxo_utils" package
    ntx: int
    nrx: int


def load_sensor_info(sensor_name):

    if sensor_name.lower() == "ultratem":
        Sensor.ntx = 5
        Sensor.nrx = 11
    elif sensor_name.lower() == "ultratema":
        Sensor.ntx = 4
        Sensor.nrx = 12

    return Sensor


# normalize data across by the max amplitude of all transmitter-receiver combinations:
def normalize_data(d, eps=1e-6): 
    dd = d.reshape(d.shape[0], np.prod(d.shape[1:]), order="F")
    normalize_by = np.max(np.abs(dd), 1) + eps # make sure we don't divide by zero
    return (dd.T/normalize_by).T.reshape(d.shape, order="F")

# re-order according to (# samples, ntx*3, nrx, ncycles, time_channels)
def data3dorder(dataset, sensor_type):
    nsamps = dataset.shape[0]
    ncycles = dataset.shape[2]
    ntimes = dataset.shape[3]

    if sensor_type.lower() == 'ultratem':
        ntx, nrx = 5, 11
        rec_ord = [0,6,1,7,2,8,3,9,4,10,5] # UltraTEM
    elif sensor_type.lower() == 'ultratema':
        ntx, nrx = 4, 12
        rec_ord = [0,1,2,3,4,5,6,7,8,9,10,11] # UltraTEMA

    d_3d = np.zeros((nsamps,ntx*3,nrx,ncycles,ntimes))

    for s in range(nsamps):
        for i in range(ntx):
            for k in range(3):
                for j,ro in enumerate(rec_ord):
                    d_3d[s,i*3+k,j,:,:] = dataset[s,ro+i*(3*nrx)+k*nrx,:,:]
                    
    return d_3d

def get_coords_data3d(net_x_profile, net_y_profile, nsteps, ncycles, step_size, sensor_type):
    L = nsteps*ncycles - (ncycles-step_size)*(nsteps-1)

    if sensor_type.lower() == 'ultratem':
        ntx, nrx = 5, 11
        rec_ord = [0,6,1,7,2,8,3,9,4,10,5] # UltraTEM
    elif sensor_type.lower() == 'ultratema':
        ntx, nrx = 4, 12
        rec_ord = [0,1,2,3,4,5,6,7,8,9,10,11] # UltraTEMA

    x_seg = np.zeros((nrx,L))
    y_seg = np.zeros((nrx,L))
    for i in range(nsteps-1):
        x_seg[:,i*step_size:(i+1)*step_size] = net_x_profile[i,rec_ord,:step_size]
        y_seg[:,i*step_size:(i+1)*step_size] = net_y_profile[i,rec_ord,:step_size]
        #print(i)
    x_seg[:,-ncycles:] = net_x_profile[-1,0:nrx,:]
    y_seg[:,-ncycles:] = net_y_profile[-1,0:nrx,:]
    
    return x_seg,y_seg

