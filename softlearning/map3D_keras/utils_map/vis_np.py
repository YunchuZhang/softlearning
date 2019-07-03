from . import binvox_rw
import numpy as np


def binarize(a, thres=0.5):
  a[a>thres] = 1
  a[a<thres] = 0
  return a 

def save_voxel(voxel_, filename, THRESHOLD=0.5):
  S1 = voxel_.shape[2]
  S2 = voxel_.shape[1]
  S3 = voxel_.shape[0]

  binvox_obj = binvox_rw.Voxels(
    np.transpose(voxel_, [2, 1, 0]) >= THRESHOLD,
    dims = [S1, S2, S3],
    translate = [0.0, 0.0, 0.0],
    scale = 1.0,
    axis_order = 'xyz'
  )   

  with open(filename, "wb") as f:
    binvox_obj.write(f)
