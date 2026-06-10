import numpy as np
import h5py
import os
from imageio import imsave
from scipy.io import loadmat
# from TriFormer.utils.imresize import *
from utils.imresize import *

# sourceDataPath = '/SRdata/lhsDATA/sourcedata'
sourceDataPath = '/SRdata/lhsDATA/dataset-NTIRE'  #
sourceDatasets = os.listdir(sourceDataPath)

# resultsPath = '/SRdata/lhsDATA/all_LFresult/mat_result/LFMT1.71/SR_5x5_4x/'
# SavePath = '/SRdata/lhsDATA/all_LFresult/SRimage/LFMT1.71/SR_5x5_4x/'

# resultsPath = '/SRdata/lhsDATA/all_LFresult/mat_result/LFMT1.71/SR_5x5_4x/'
# SavePath = '/SRdata/lhsDATA/all_LFresult/SRimage/LFMT1.71/SR_5x5_4x/'
resultsPath = '/SRdata/lhsDATA/bs/LFMT*1.71/SR_5x5_4x/'
SavePath = '/SRdata/lhsDATA/HRS1'
angRes = 5

def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]
    return y

for datasetName in sourceDatasets:

    # gtFolder = os.path.join(sourceDataPath, datasetName, 'test') 
    gtFolder = os.path.join(sourceDataPath, datasetName, 'inference') 
    sceneFiles = os.listdir(gtFolder)
    #resultsFolder = os.path.join(resultsPath, datasetName)
    resultsFolder = os.path.join(resultsPath, datasetName)
    for scene in sceneFiles:
        sceneName = scene.split('.')[0]
        print(f'Generating result images of Scene_{sceneName} in Dataset {datasetName} ......')
        gtPath = os.path.join(gtFolder, scene)
        resultPath = os.path.join(resultsFolder, sceneName + '.mat')

        try:
            data = h5py.File(gtPath, 'r')
            LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
        except:
            data = loadmat(gtPath)
            LF = np.array(data[('LF')])

        try:
            data = h5py.File(resultPath, 'r')
            outLF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
        except:
            data = loadmat(resultPath)
            outLF = np.array(data[('LF')])

        (U, V, H, W) = outLF.shape
        print(LF.shape, outLF.shape)
        # Extract central angRes * angRes views
        LF = LF[(U - angRes) // 2:(U + angRes) // 2, (V - angRes) // 2:(V + angRes) // 2, :H, :W, 0:3]
        LF = LF.astype('double')

        for u in range(U):
            for v in range(V):
                hr_SAI = LF[u, v, :, :, :]
                sr_sai_y = outLF[u, v, :, :]
                hr_SAI_CbCr = rgb2ycbcr(hr_SAI)[:, :, 1:3]
                hr_SAI_CbCr = imresize(hr_SAI_CbCr, scalar_scale=4) #新增
                hr_sai_ycbcr = np.concatenate([sr_sai_y.reshape(H, W, -1), hr_SAI_CbCr], axis=-1)
                hr_sai_grb = ycbcr2rgb(hr_sai_ycbcr)
                savePath = os.path.join(SavePath, datasetName, sceneName)
                os.makedirs(savePath, exist_ok=True)
                data = np.array(hr_sai_grb.clip(0, 1) * 255, dtype='uint8')
                imsave(os.path.join(savePath, 'View_' + str(u) + '_' + str(v) + '.bmp'), data)