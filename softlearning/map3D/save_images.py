import os
from scipy.misc import imsave
import numpy as np
def save_replay_buffer(fields):
  # key_val = fields.keys()
  for i in range(400):
    images = fields["observations.image_observation"][i]
    depths = fields["observations.depth_observation"][i]
    angles = fields["observations.cam_angles_observation"][i]
    img_folder_name = "data/images/"+str(i)
    depth_folder_name = img_folder_name.replace("images","depths")
    try:
      os.makedirs(img_folder_name)
    except Exception:
      print("file exists")
    try:
      os.makedirs(depth_folder_name)
    except Exception:
      print("file exists")  
    for view in range(54):
      image_view  = images[view]
      depth_view  = depths[view]/2
      elevation,azimuth = angles[view]
      file_name = "{}_{}.png".format(azimuth,elevation)
      image_name = img_folder_name + "/" + file_name
      depth_name = depth_folder_name + "/" + file_name.replace("png","npy")
      # st()
      imsave(image_name,image_view)
      np.save(depth_name,depth_view)
      # imsave(depth_name,depth_view)