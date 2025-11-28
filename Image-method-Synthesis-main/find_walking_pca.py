import numpy as np
import copy
import json
import math
from glob import glob
import scipy.spatial.distance as sciDist
from tqdm import tqdm
import requests
import time
import itertools
import random
import os
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
from itertools import islice
from PIL import Image
import re
from tqdm import tqdm


# Headless simulator version
index = 7 # local server index 
#API_ENDPOINT = 'http://localhost:400' + str(index) + '/simulation' # NOT THE LS VERSION
API_ENDPOINT = 'http://localhost:400' + str(4) + '/simulation' # NOT THE LS VERSION
HEADERS = {"Content-Type": "application/json"}
batchCount = 25 # Send this number of samples to MotionGen each time 
speedscale = 1
steps = 360
minsteps = int(steps*20/360)

# Search for "stephenson" or "watt" in the JSON file and collect their names and indices
json_path = "KV_468_062324.json"
with open(json_path, "r") as f:
    data = json.load(f)

matches = []
for idx, name in enumerate(data):
    lname = name.lower()
    if "step" in lname or "watt" in lname:
        matches.append([name, idx])
print(matches)


# Things for 4 bar 
mechType = index
types = ['RRRR', 'RRRP', 'RRPR', 'PRPR']
for mechanism in matches:
    types.append(mechanism[0])
print(types)
# ['RRRR', 'RRRP', 'RRPR', 'PRPR', 'Watt2T1A1', 'Steph1T3', 'Watt1T1A1', 'Steph3T2A2', 'Watt1T3A2', 'Steph2T2A1', 'Steph3T1A2', 'Watt2T2A2', 'Watt1T2A1', 'Steph2T1A1', 'Steph3T2A1', 'Watt1T1A2', 'Watt2T1A2', 'Steph1T2', 'Steph1T1', 'Watt1T3A1', 'Watt1T2A2', 'Watt2T2A1', 'Steph3T1A1', 'Steph2T1A2', 'Steph2T2A2']

# Create a directory in outputs-6bar for each type in types
'''
for t in types:
    if t in ('RRRR', 'RRRP', 'RRPR', 'PRPR'):
        continue
    dir_path = os.path.abspath("./outputs-6bar/" + t)
    os.makedirs(dir_path, exist_ok=True)'''



typeIndex = [49, 64, 155, 175] # to avoid confusion from any other type
for mechanism in matches:
    typeIndex.append(mechanism[1])

print(typeIndex)
# [49, 64, 155, 175, 0, 2, 19, 26, 37, 39, 73, 82, 90, 110, 111, 125, 133, 135, 137, 139, 149, 163, 169, 176, 177]


#couplerCurveIndex = [4, 4, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]


#confirm that all c values are 1 at index 7 for 6 bars

bsi = "BSIdict_468_062324_3.json"
with open(bsi, "r") as b:
    data = json.load(b)

for key in types:
    if key in data:
        c_value = (data)[key].get("c")
        print(key, c_value)

# Update couplerCurveIndex so that for each type, its value is the index of the number 1 in the c_value list from BSIdict
couplerCurveIndex = []
for key in types:
    if key in data:
        c_value = data[key].get("c")
        if isinstance(c_value, list) and 1 in c_value:
            couplerCurveIndex.append(c_value.index(1))
        else:
            couplerCurveIndex.append(4)  # fallback default
    else:
        couplerCurveIndex.append(4)      # fallback default
print("Updated couplerCurveIndex:", couplerCurveIndex)
# Updated couplerCurveIndex: [4, 4, 4, 4, 7, 7, 7, 7, 6, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 4, 7, 4]


savePointNumber = [5, 6, 7, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8] #len of c arrat acording to BSIdict
needAddtional = [False, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] # notice that when true, some of the points can change its position randomly 
initStates = np.load("./npy-inputs/" + 'Randpos-.npy')
errCtr = 0
batch = []
batchSaveStr = []
batchSaveNpyStr = []


# The transformation 
#np.save(saveDir + name + ' ' + types[index], param)
saveDir = os.path.abspath("./outputs-6bar/" + types[index] )
saveDirNpy = os.path.abspath("./outputs-6bar/" + types[index] + "-npy")
print(saveDir, saveDirNpy)

# good old ones 

def isValid(seq):
    if len(seq.shape) == 2:
        isVal = np.var(seq[:,0]) <= 5e-3 and np.var(seq[:,1]) <= 5e-3
    else:
        isVal = len(seq) == 0 or np.var(seq) <= 5e-3

    if isVal:
        return False
    else:
        return True


def get_pca_inclination(qx, qy, ax=None, label=''):
    """ Performs the PCA
        Return transformation matrix
    """
    cx = np.mean(qx)
    cy = np.mean(qy)
    covar_xx = np.sum((qx - cx)*(qx - cx))/len(qx)
    covar_xy = np.sum((qx - cx)*(qy - cy))/len(qx)
    covar_yx = np.sum((qy - cy)*(qx - cx))/len(qx)
    covar_yy = np.sum((qy - cy)*(qy - cy))/len(qx)
    covar = np.array([[covar_xx, covar_xy],[covar_yx, covar_yy]])
    eig_val, eig_vec= np.linalg.eig(covar)

    # Inclination of major principal axis w.r.t. x axis
    if eig_val[0] > eig_val[1]:
        phi= np.arctan2(eig_vec[1,0], eig_vec[0,0])
    else:
        phi= np.arctan2(eig_vec[1,1], eig_vec[0,1])

    return phi


def get_normalize_curve(jd, steps=None, rotations=1, normalize=True, transformParas=None):
    jd = np.array(jd)
    joint_data_n, x_mean, y_mean, denom, phi = [], None, None, None, None
    if isValid(jd):
        if steps:
            sample_indices = np.linspace(0, jd.shape[0]-1, steps, dtype=np.int32)
            jd = jd[sample_indices,:]
        if normalize:
            if not transformParas:
                x_mean = np.mean(jd[:,0], axis=0, keepdims=True)
                y_mean = np.mean(jd[:,1], axis=0, keepdims=True)
            else:
                x_mean, y_mean, denom, phi = transformParas
            jd[:,0] = jd[:,0] - x_mean
            jd[:,1] = jd[:,1] - y_mean

            if not transformParas:
                denom = np.sqrt(np.var(jd[:,0], axis=0, keepdims=True) + np.var(jd[:,1], axis=0, keepdims=True))
                denom = np.expand_dims(denom, axis=1)
            jd = jd / denom
            t = 0
        if not transformParas:
            phi = -get_pca_inclination(jd[:,0], jd[:,1])
        jd[:,0], jd[:, 1] = rotate_curve(jd, phi)
        for tt in range(rotations):
            joint_data_n.append(jd.copy())
            if rotations > 1:
                jd[:,0], jd[:,1] = rotate_curve(jd, t)
                t = 2*np.pi/rotations

    return joint_data_n, x_mean, y_mean, denom, phi


def rotate_curve(cur, theta):
    cpx = cur[:,0]*np.cos(theta) - cur[:,1]*np.sin(theta)
    cpy = cur[:,0]*np.sin(theta) + cur[:,1]*np.cos(theta)
    return cpx, cpy


def digitize_seq(nums, minlim, maxlim, bin_size=64):
    bins = np.linspace(minlim, maxlim, bin_size-1)
    nums_indices = np.digitize(nums, bins)
    return nums_indices


def get_normalize_joint_data_wrt_one_curve(joint_data, ref_ind = 4):
    ''' input s = [num_curves, num_points, 2]
    '''
    joint_data_n = []
    s = np.array(joint_data)
    if isValid(s[ref_ind]):
        x_mean = np.mean(s[ref_ind:ref_ind+1,:,0], axis=1, keepdims=True)
        y_mean = np.mean(s[ref_ind:ref_ind+1,:,1], axis=1, keepdims=True)
        s[:,:,0] = s[:,:,0] - x_mean
        s[:,:,1] = s[:,:,1] - y_mean
        denom = np.sqrt(np.var(s[ref_ind:ref_ind+1,:,0], axis=1, keepdims=True) + np.var(s[ref_ind:ref_ind+1,:,1], axis=1, keepdims=True))
        denom = np.expand_dims(denom, axis=2) #is this scale? 
        s = s / denom
        phi = -get_pca_inclination(s[ref_ind:ref_ind+1,:,0], s[ref_ind:ref_ind+1,:,1])
        for i in range(s.shape[0]):
            s[i,:,0], s[i,:,1] = rotate_curve(s[i], phi)
    else:
        return s, [None, None, None, None], False

    # s has a shape of (j_num, state, dim)
    return s, [x_mean[0][0], y_mean[0][0], denom[0][0][0], phi], True # tx, ty, scaling, rotation angle 


##############################################################################################
# There are some other necessary transformations. (x_mean, y_mean, phi, denom) are from get_normalize_curve. 
##############################################################################################
def get_image_from_point_cloud(points, xylim, im_size, inverted = True, label=None):
    mat = np.zeros((im_size, im_size, 1), dtype=np.uint8)
    x = digitize_seq(points[:,0], -xylim, xylim, im_size)
    if inverted:
        y = digitize_seq(points[:,1]*-1, -xylim, xylim, im_size)
        mat[y, x, 0] = 1
    else:
        y = digitize_seq(points[:,1], -xylim, xylim, im_size)
        mat[x, y, 0] = 1
    return mat


def process_mech_102723(jointData, ref_ind, im_size = 64, xylim = 3.5, inverted = True, swapAxes = True):
    paras = None

    # It is possible the jointData format is (angles, joint, (x, y)). 
    # You should put a True if this happens. (This is how files are saved).
    # I literally don't understand why I saved jointData with a shape of (angles, joint, (x, y)) 
    if swapAxes:
        jointData = np.swapaxes(jointData, 0, 1)

    # This converts all 
    jointData, paras, success = get_normalize_joint_data_wrt_one_curve(jointData, ref_ind= ref_ind)

    # jointData format from now on becomes np.array with a shape of (joint, curve_length, dimension)
    jointData = np.array(jointData)

    if success:
        # get binaryImage 
        jd = jointData[ref_ind]
        mat = get_image_from_point_cloud(jd, xylim=xylim, im_size=im_size, inverted=inverted)
        return mat, paras, success
    else: 
        return None, None, success


def calc_dist(coord):
    # Calculate differences using broadcasting
    diffs = coord[:, np.newaxis, :] - coord[np.newaxis, :, :]
    squared_dists = np.sum(diffs ** 2, axis=2)

    # Extract the upper triangle indices where i < j
    i, j = np.triu_indices(len(coord), k=1)
    dist_arr = np.sqrt(squared_dists[i, j])
    dist_arr = dist_arr/min(dist_arr)
    return np.round(dist_arr, 2)