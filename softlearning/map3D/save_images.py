import os
from scipy.misc import imsave
import numpy as np
def save_replay_buffer(fields):
  # key_val = fields.keys()
  for i in range(4):
    images = fields["observations.image_observation"][i]
    depths = fields["observations.depth_observation"][i]
    angles = fields["observations.cam_angles_observation"][i]
    img_folder_name = "data/images/"+str(i)
    depth_folder_name = img_folder_name.replace("images","depths")
    env_sample_images = "env_data"
    try:
      os.makedirs(img_folder_name)
    except Exception:
      print("file exists")
    try:
      os.makedirs(depth_folder_name)
    except Exception:
      print("file exists")
    try:
      os.makedirs(env_sample_images)
    except Exception:
      print("file exists")  
    for view in range(4):
      image_view  = images[view]
      depth_view  = depths[view]
      elevation,azimuth = angles[view]
      file_name = "{}_{}.png".format(azimuth,elevation)
      image_name = img_folder_name + "/" + file_name
      depth_name = depth_folder_name + "/" + file_name.replace("png","npy")
      # st()
      imsave(image_name,image_view)
      np.save(depth_name,depth_view)
      # imsave(depth_name,depth_view)
def save_some_samples(sampler):
  observation_keys_o = ["observations.image_observation","observations.depth_observation","observations.image_desired_goal","observations.desired_goal_depth","observations.achieved_goal"]
  for i_num in range(3):
    obs = sampler.random_batch()
    for i_key in observation_keys_o:
      curr_ob = obs[i_key][0]
      arr = np.vsplit(curr_ob,curr_ob.shape[0])
      for i,val in enumerate(arr):
        print(val.shape)
        # st()
        # elevation,azimuth = obs[observation_keys_o[2]][0][i]
        # st()
        # camera.azimuth = angle_range / (n - 1) * i + start_angle
        # camera.elevation = start_angle + angle_delta*angle_i
        # azimuth =  start_angle - angle_delta*i
        imsave("env_data/batch_{}_{}_{}_angle.png".format(i_num,i_key,i),val[0])