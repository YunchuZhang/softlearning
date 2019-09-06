import sys
import math
import time
from softlearning import __init__path
from options import OptionGroup as OG
import options

opname = 'reconstruction'
net_name = "3d"
eager = False
run_full = False
# unlikely to change constants:
NUM_VIEWS = 3
NUM_PREDS = 1

MULTI_UNPROJ = True
AGGREGATION_METHOD = 'stack'
AUX_POSE = True
DUMP_TENSOR = False

PHI_IDX = None
CATS = 57 #number of categories
eps = 1E-8
#V = 18  # number of views for multiview -- we are keeping phi frozen for now

Hdata = 128
Wdata = 128

inject_summaries = False
summ_grads = False

HV = 18
VV = 3
MINH = 0
MAXH = 360 #exclusive
MINV = 0
MAXV = 30 #exclusive
HDELTA = (MAXH-MINH) / HV #20
VDELTA = (MAXV-MINV) / VV #10

H = 256
W = 256
anchor_size = 0.25 #0.25 #[0.25, 0.25, 0.25]
#anchor_size = [1.0, 1.0, 1.0]

fov = 30.0
radius = 4.0
boundary_to_center=1.0
sigma = 0.05 # for visualization
S = 128  # cube size
BS = 2
SS = 16
NS = BS * SS
NB_STEPS = 1000000


fs_2D = 32
ORIENT = True
STNET = False
# for rpn
crop_size=16
crop_feat_size=0.125
P_THRES = 0.5
TRAIN_ROIS_PER_IMAGE = 10
ROI_POSITIVE_RATIO = 0.3
ROI_IOU_MAX=0.5
ARCH = 'unproj'
CROP_SIZE=16
#options: 'unproj, marr'

# dynamics:
max_T = 15
rollout_T = 5

NET3DARCH = 'marr' #or '3x3, marr'
USE_OUTLINE = True
USE_MESHGRID = True
USE_LOCAL_BIAS = False #set to false later

INPUT_RGB = False
INPUT_POSE = False
VOXNET_LATENT = 512
VIS_OUT_SIZE = 128
IS_DUMP_VIS=True
# test/train mode
mode = 'train'

# input constants
train_file = 'all'
val_file = 'all'
test_file = 'all'
split_format=False
split_cut=False
is_trainval_diff_summ = False
# optimizer consts
lr = 1E-4
mom = 0.9

# validation period
valp = 1000
savep = 2000

# important directories
vis_dir = 'vis'
tb_dir = 'log'
data_dir = 'data'
ckpt_dir = 'ckpt'
ckpt_cfg_dir = 'ckpt_cfg'
LOSS_FN="CE"


# debug flags
FAKE_NET = False
REPROJ_SINGLE = False
ADD_FEEDBACK = False
VALIDATE_INPUT = False
DEBUG_MODE = False
DEBUG_32 = False
DEBUG_HISTS = False
DEBUG_PLACEMENT = False
DEBUG_VOXPROJ = False
DEBUG_VOXNET = False
DEBUG_REPROJ = False
DEBUG_EXPORTS = True
DEBUG_SPEED = True
DEBUG_NET = False
DEBUG_RP = False
DEBUG_FULL_TRACE = False
DEBUG_NODE_TRACE = False
DEBUG_NAN = False
DEBUG_CACHE = False
DEBUG_LOSSES = True
DEBUG_MEMORY = False
DEBUG_UNPROJECT = False
DEBUG_SAME_BATCH=False


SKIP_RUN = False
SKIP_TRAIN_EXPORT = False
SKIP_VAL_EXPORT = False
SKIP_EXPORT = False

USE_GRAVITY_LOSS = False

FIX_VIEW = False
STOP_PRED_DELTA = True
STOP_REPROJ_MASK_GRADIENTS = False

USE_TIMELINE = False

rpvx_unsup = False
force_batchnorm_trainmode = False
force_batchnorm_testmode = False

RANDOMIZE_BG = True

MNIST_CONVLSTM = False
MNIST_CONVLSTM_STOCHASTIC = False

GQN3D_CONVLSTM = False
GQN3D_CONVLSTM_STOCHASTIC = False

EGO_PRETRAIN=False
DEPTH_PRETRAIN=False
USE_PREDICTED_DEPTH=False
USE_GT_ROIS = False
BIN_THRES = 0.5

# for bullet pushing task
USE_PREDICTED_EGO=False
USE_ORN_IN_FIRST_STEP=False
IS_PREDICT_CONTACT = False
PRETRAIN_CONTACT=False
ARCH = ""
BBOX_RANDOMIZATION=False
OUTLINE_PRECOMPUTED = False
BN_DECAY=0.9
IS_NOT_BN_IN_2D = False
IS_NOT_BN_IN_3D = False
IS_NOT_BN_IN_FLOW = False,
AGENT_WITHOUT_GLOBAL=False
MASK_AGENT=False
USE_3D_REFINE=True
USE_AGENT_GT=False
DUMP_GRAPH=False

train_vp = False
train_on_val = False

IS_VIEW_PRED = train_vp
ckpt_base = ""
detector= False
####

T0 = time.time()

exp_name = sys.argv[1].strip() if len(sys.argv) >= 2 else ''
run_name = sys.argv[2].strip() if len(sys.argv) >= 3 else '1'



load_name = ''
save_name = exp_name + "/" + run_name

options.data_options('doubledata', 'double', add_suffix = True)
options.data_options('xian_doubledata', 'xian_doubledata', add_suffix = True)
options.data_options('realsense_data', 'realsense', add_suffix = True)
options.data_options('double130data', 'double_130', add_suffix = True)
options.data_options('double110data', 'double_110', add_suffix = True)
options.data_options('double910data', 'double_910', add_suffix = True)
options.data_options('double15data', 'double_15', add_suffix = True)
options.data_options('double45data', 'double_45', add_suffix = True)
options.data_options('doubledebugdata', 'double_single', add_suffix = False)

options.data_options("bulletpushdata", "bulletpush", add_suffix=True)
options.data_options("bulletpushdata2", "bulletpush2", add_suffix=True)
options.data_options("bulletpushdata_airplane", "bulletpush_airplane", add_suffix=True)
options.data_options("bulletpushdata_bags", "bulletpush_bags", add_suffix=True)
options.data_options("bulletpushdata_bags_1", "bulletpush_bags_1", add_suffix=True)
options.data_options("bulletpushonedata", "bulletpush_one", add_suffix=True)

options.data_options("bulletpushdata_basic", "bulletpushdata_basic", add_suffix=True)


options.data_options("bulletpushdata_basic_ht", "bulletpushdata_basic_ht", add_suffix=True)
options.data_options("bulletpushdata_basic_ht1", "bulletpushdata_basic_ht1", add_suffix=True)
# this data set use the heuristic push, the test data contains same action/state push
# on different objects
options.data_options("bulletpushdata_basic_h1", "bulletpushdata_basic_h1", add_suffix=True)
# this data has larger training dataset
options.data_options("bulletpushdata_basic_h2", "bulletpushdata_basic_h2", add_suffix=True)

options.data_options("bulletpushdata_mitpush_same_200", "bulletpushdata_mitpush_same_200", add_suffix=True)
options.data_options("bulletpushdata_0609_t5_200", "bulletpushdata_0609_t5_200", add_suffix=True)


options.data_options("bulletpushdata_mitpush_same_200_split", "bulletpushdata_mitpush_same_200_split", add_suffix=True, split_format=True)
options.data_options("bulletpushdata_mitpush_same_200_split_debug", "bulletpushdata_mitpush_same_200_split_debug", add_suffix=True, split_format=True)

options.data_options("bulletpushdata_mitpush_same_200_split_debug1", "bulletpushdata_mitpush_same_200_split_debug1", add_suffix=True, split_format=True, split_cut=1)


options.data_options("bulletpushdata_0520_val_5steps_20", "bulletpushdata_0520_val_5steps_20", add_suffix=True, split_format=True)


# xian data
options.data_options("bulletpushdata_cotrain", "bulletpushdata_cotrain", add_suffix=True)
options.data_options("bulletpushdata_200_debug", "bulletpushdata_200_debug", add_suffix=True)

options.data_options("mujocopickplace", "mujocopickplace", add_suffix=True)

OG("bulletpush",
   "bulletpushdata_basic_h2",
   lr=0.01,
   ARCH="convfc",
   #mode="test",
   #eager=True,
   BS=12,
   opname="bulletpush", H = 128, W=128, data_dir="/projects/katefgroup/fish/dynamics/",
   #data_dir="/projects/katefgroup/xian/data",
   #eager=True,
   USE_ORN_IN_FIRST_STEP=True,
   boundary_to_center=1.5, valp=1000, is_trainval_diff_summ=True)
   #load_name = "bulletpush/0509_lr2_h2") #, lr=1E-8)

OG("bulletpush_debug_bn",
   #"bulletpushdata_mitpush_same_200",
   "bulletpushdata_200_debug",
   lr=0.01,
   #mode="test",
   ARCH="convfc",
   #mode="test",
   #eager=True,
   BS=8,
   opname="bulletpush", H = 128, W=128, data_dir="/projects/katefgroup/fish/dynamics/",
   #eager=True,
   USE_ORN_IN_FIRST_STEP=True,
   boundary_to_center=1.5, valp=100, is_trainval_diff_summ=True,
   #load_name = "bulletpush_debug_bn/bad_slim_with_update")
   )


OG("0517_bulletpush",
   "bulletpushdata_mitpush_same_200",
   IS_DUMP_VIS=False,
   mode="test",
   #"bulletpushdata_cotrain",
   lr=0.001,
   #mode="test",
   ARCH="convfc",
   #eager=True,
   BS=8,
   opname="bulletpush", H = 128, W=128,
   data_dir="/projects/katefgroup/fish/dynamics/",
   #data_dir="/projects/katefgroup/xian/data",
   #eager=True,
   USE_ORN_IN_FIRST_STEP=True,
   boundary_to_center=1.5, valp=100, is_trainval_diff_summ=True)
   #load_name = "bulletpush_debug_bn/bad_slim_with_update")





OG("0517_bulletpush_test",
   "bulletpushdata_0520_val_5steps_20",
   #"bulletpushdata_mitpush_same_200",
   #"bulletpushdata_cotrain",
   #IS_DUMP_VIS=False,
   lr=0.001,
   mode="test",
   ARCH="convfc",
   USE_AGENT_GT=True,
   #eager=True,
   BS=1,
   opname="bulletpush", H = 128, W=128,
   data_dir="/projects/katefgroup/fish/dynamics/",
   #data_dir="/projects/katefgroup/xian/data",
   #eager=True,
   USE_ORN_IN_FIRST_STEP=True,
   boundary_to_center=1.5, valp=100, is_trainval_diff_summ=True,
   load_name="0517_bulletpush/same200_lr3_bs8_fast")


OG("0517_bulletpush_dump_graph",
   "bulletpushdata_0520_val_5steps_20",
   #"bulletpushdata_mitpush_same_200",
   #"bulletpushdata_cotrain",
   #IS_DUMP_VIS=False,
   lr=0.001,
   mode="test",
   ARCH="convfc",
   USE_AGENT_GT=True,
   DUMP_GRAPH=True,
   #eager=True,
   BS=1,
   opname="bulletpush", H = 128, W=128,
   data_dir="/projects/katefgroup/fish/dynamics/",
   #data_dir="/projects/katefgroup/xian/data",
   #eager=True,
   USE_ORN_IN_FIRST_STEP=True,
   boundary_to_center=1.5, valp=100, is_trainval_diff_summ=True,
   load_name="0517_bulletpush/same200_lr3_bs8_fast")

OG("bulletpush_debug",
   "bulletpushdata_basic_ht",
   lr=0.01,
   ARCH="convfc",
   #eager=True,
   BS=12,
   opname="bulletpush", H = 128, W=128, data_dir="/projects/katefgroup/fish/dynamics/",
   #eager=True,
   USE_ORN_IN_FIRST_STEP=True,
   boundary_to_center=1.5, valp=1000, is_trainval_diff_summ=True)
   #load_name = "bulletpush/0509_lr2_h2") #, lr=1E-8)


OG("bulletpush3D",
   #"bulletpushdata_basic",
   "bulletpushdata_basic_h2",
   #eager=True,
   lr = 0.001,
   opname="bulletpush3D", H =64, W=64, data_dir="/projects/katefgroup/fish/dynamics",
   AGGREGATION_METHOD = 'gru', USE_MESHGRID=False,
   crop_size=16,
   #eager=True,
   max_T = 1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   DEBUG_UNPROJECT=False, BS=2, valp=500, is_trainval_diff_summ=True,
   #load_name="bulletpush3D/test_lowres16",
)

OG("bulletpush3D_test",
   "bulletpushdata_mitpush_same_200",
   #"bulletpushdata_basic_h2",
   DEBUG_LOSSES=False,
   IS_DUMP_VIS=False,
   mode="test",
   lr = 0.001,
   opname="bulletpush3D", H =64, W=64, data_dir="/projects/katefgroup/fish/dynamics",
   AGGREGATION_METHOD = 'gru', USE_MESHGRID=False,
   crop_size=16,
   #eager=True,
   max_T = 1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   DEBUG_UNPROJECT=False, BS=2, valp=500, is_trainval_diff_summ=True,
   load_name="bulletpush3D/lr3_angle_l1_loss",
)

OG("bulletpush3D_2d",
   "bulletpushdata_basic_h2", #_airplane",
   #eager=True,
   #mode="test",
   lr = 0.01,
   opname="bulletpush3D_2d", H =64, W=64, data_dir="/projects/katefgroup/fish/dynamics",
   AGGREGATION_METHOD = 'gru', USE_MESHGRID=False,
   crop_size=32,
   crop_feat_size=0.125,
   #eager=True,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   DEBUG_UNPROJECT=False, BS=12, valp=500, is_trainval_diff_summ=True,
   load_name="bulletpush3D_2d/debug_h2_lr3",
   #load_name="bulletpush3D_2d/0508_full_state_lr3_3conv_conti"

)

OG("bulletpush3D_2d_debug",
   "bulletpushdata_basic_h2", #_airplane",
   #eager=True,
   #mode="test",
   lr = 0.01,
   opname="bulletpush",
   H =64, W=64, data_dir="/projects/katefgroup/fish/dynamics",
   AGGREGATION_METHOD = 'gru', USE_MESHGRID=False,
   crop_size=32,
   crop_feat_size=0.125,
   #eager=True,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   DEBUG_UNPROJECT=False, BS=12, valp=500, is_trainval_diff_summ=True,
   #load_name="bulletpush3D_2d_debug/debug_h2_lr3",
   #load_name="bulletpush3D_2d/0508_full_state_lr3_3conv_conti"
)

OG("bulletpush3D_3",
   "bulletpushdata_bags", #_airplane",
   #eager=True,
   lr=0.001,
   opname="bulletpush3D_3", H =64, W=64, data_dir="/projects/katefgroup/fish/dynamics/push_0409/",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=16,
   #eager=True,
   max_T=10,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   DEBUG_UNPROJECT=False,
   BS=2, valp=500, is_trainval_diff_summ=True,
   #load_name="bulletpush3D_3/test_lr3"
)

OG("bulletpush3D_4_multicam",
   "bulletpushdata_mitpush_same_200",
   #"bulletpushdata_basic_ht",
   eager=True,
   #DEBUG_LOSSES=False,
   #IS_DUMP_VIS=False,
   #PRETRAIN_CONTACT=True,
   #IS_PREDICT_CONTACT=True,
   OUTLINE_PRECOMPUTED = True,
   USE_OUTLINE=False,
   lr=0.001,
   opname="bulletpush3D_4", H =64, W=64,
   data_dir="/projects/katefgroup/fish/dynamics/",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=32,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   #DEBUG_UNPROJECT=True,
   BS=2, valp=500, is_trainval_diff_summ=True,
   #load_name = "bulletpush3D_4_multicam/bs8_contact")
   )

OG("0517_bulletpush3D_4_multicam_bn",
   "bulletpushdata_mitpush_same_200_split",
   #"bulletpushdata_mitpush_same_200",
   #BN_DECAY=0.7,
   #eager=True,
   #"bulletpushdata_cotrain",
   #"bulletpushdata_basic_ht",
   #DEBUG_LOSSES=False,
   #IS_DUMP_VIS=False,
   #PRETRAIN_CONTACT=True,
   #IS_PREDICT_CONTACT=True,
   #OUTLINE_PRECOMPUTED = True,
   #USE_OUTLINE=False,
   #AGENT_WITHOUT_GLOBAL=True,
   #IS_NOT_BN_IN_2D = True,
   IS_NOT_BN_IN_3D = True,
   lr=0.001,
   opname="bulletpush3D_4", H =64, W=64,
   data_dir="/projects/katefgroup/fish/dynamics/",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=16,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   #DEBUG_UNPROJECT=True,
   BS=2, valp=500, is_trainval_diff_summ=True,
   #load_name = "0517_bulletpush3D_4_multicam_bn/no_bn_lr3"
   #load_name = "0517_bulletpush3D_4_multicam_bn/lr3_bs2_kill_batchnorm_on_top"
   #load_name = "0517_bulletpush3D_4_multicam_bn/0518_shuffle_no_bn_on_top_and_3d_lr3_bs2"
   )

# OG("audrey_test",
#    "mujocopickplace",
#    #"bulletpushdata_cotrain",
#    #"bulletpushdata_mitpush_same_200_split",
#    #IS_DUMP_VIS=False,
#    MASK_AGENT=False, 
#    eager=True,
#    #OUTLINE_PRECOMPUTED = True,
#    #USE_OUTLINE=True,
#    #IS_NOT_BN_IN_2D = True,
#    IS_NOT_BN_IN_3D = True,
#    lr=0.000001,
#    opname="bulletpush3D_4", H =64, W=64,
#    #data_dir="/projects/katefgroup/xian/data",
#    data_dir="/projects/katefgroup/audrey/dynamics",
#    AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
#    crop_size=32,
#    max_T=1,
#    DEBUG_UNPROJECT=True,
#    fs_2D=8,
#    BS=4, valp=500, is_trainval_diff_summ=True,
#    radius = 0.8, boundary_to_center=0.9, fov = 110
#    )

OG("0517_bulletpush3D_4_multicam_bn_test",
   "bulletpushdata_mitpush_same_200_split",
   mode="test",
   IS_DUMP_VIS=False,
   IS_NOT_BN_IN_2D = False,
   IS_NOT_BN_IN_3D = True,
   lr=0.001,
   opname="bulletpush3D_4", H =64, W=64,
   data_dir="/projects/katefgroup/fish/dynamics/",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=16,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   BS=5, valp=500, is_trainval_diff_summ=True,
   load_name = "0517_bulletpush3D_4_multicam_bn/no_bn_on_top_3d_shuffle_lr5_conti"
   )


OG("0517_bulletpush3D_4_multicam_bn_mask",
   "bulletpushdata_mitpush_same_200",
   #"bulletpushdata_cotrain",
   #"bulletpushdata_mitpush_same_200_split",
   #IS_DUMP_VIS=False,
   MASK_AGENT=True,
   #OUTLINE_PRECOMPUTED = True,
   #USE_OUTLINE=True,
   #IS_NOT_BN_IN_2D = True,
   #IS_NOT_BN_IN_3D = True,
   lr=0.001,
   opname="bulletpush3D_4", H =64, W=64,
   #data_dir="/projects/katefgroup/xian/data",
   data_dir="/projects/katefgroup/fish/dynamics",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=16,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   #eager=True,
   #DEBUG_UNPROJECT=True,
   BS=4, valp=500, is_trainval_diff_summ=True,
   #load_name = "0517_bulletpush3D_4_multicam_bn_mask/no_bn_on_top_3d_shuffle"
   #load_name = "0517_bulletpush3D_4_multicam_bn_mask/xian_bs4"
   )


OG("0610_bulletpush3D_4_multicam_bn_mask",
   "bulletpushdata_0609_t5_200",
   #IS_DUMP_VIS=False,
   MASK_AGENT=True,
   #OUTLINE_PRECOMPUTED = True,
   #USE_OUTLINE=True,
   #IS_NOT_BN_IN_2D = True,
   #IS_NOT_BN_IN_3D = True,
   lr=0.0001,
   opname="bulletpush3D_4", H =64, W=64,
   #data_dir="/projects/katefgroup/xian/data",
   data_dir="/projects/katefgroup/fish/dynamics",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=16,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   #eager=True,
   #DEBUG_UNPROJECT=True,
   BS=2, valp=500, is_trainval_diff_summ=True)


OG("0517_bulletpush3D_4_multicam_bn_mask_test_rollout",
   "bulletpushdata_0520_val_5steps_20",
   #"bulletpushdata_mitpush_same_200",
   #"bulletpushdata_cotrain",
   IS_DUMP_VIS=True,
   mode="test",
   MASK_AGENT=True,
   #OUTLINE_PRECOMPUTED = True,
   #USE_OUTLINE=True,
   #IS_NOT_BN_IN_2D = True,
   IS_NOT_BN_IN_3D = True,
   lr=0.000001,
   USE_AGENT_GT = True,
   opname="bulletpush3D_4", H =64, W=64,
   #data_dir="/projects/katefgroup/xian/data",
   data_dir="/projects/katefgroup/fish/dynamics",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=32,
   max_T=5,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   #DEBUG_UNPROJECT=True,
   BS=1, valp=500, is_trainval_diff_summ=True,
   load_name = "0517_bulletpush3D_4_multicam_bn_mask/no_bn_on_top_3d_shuffle_lr5_conti"
   )

OG("0517_bulletpush3D_4_multicam_bn_mask_test_rollout_dump",
   "bulletpushdata_0520_val_5steps_20",
   #"bulletpushdata_mitpush_same_200",
   #"bulletpushdata_cotrain",
   IS_DUMP_VIS=True,
   mode="test",
   MASK_AGENT=True,
   DUMP_GRAPH=True,
   #OUTLINE_PRECOMPUTED = True,
   #USE_OUTLINE=True,
   #IS_NOT_BN_IN_2D = True,
   IS_NOT_BN_IN_3D = True,
   lr=0.000001,
   USE_AGENT_GT = True,
   opname="bulletpush3D_4", H =64, W=64,
   #data_dir="/projects/katefgroup/xian/data",
   data_dir="/projects/katefgroup/fish/dynamics",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=32,
   max_T=5,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   #DEBUG_UNPROJECT=True,
   BS=1, valp=500, is_trainval_diff_summ=True,
   load_name = "0517_bulletpush3D_4_multicam_bn_mask/no_bn_on_top_3d_shuffle_lr5_conti"
   )


############################# SINGLE VIEW #########################################
OG("0520_bulletpush3D_4_multicam_bn_mask_nview1",
   "bulletpushdata_mitpush_same_200",
   #"bulletpushdata_cotrain",
   #"bulletpushdata_mitpush_same_200_split",
   #IS_DUMP_VIS=False,
   MASK_AGENT=True,
   #OUTLINE_PRECOMPUTED = True,
   #USE_OUTLINE=True,
   #IS_NOT_BN_IN_2D = True,
   NUM_VIEWS=1,
   #IS_NOT_BN_IN_3D = True,
   lr=0.0001,
   opname="bulletpush3D_4", H =64, W=64,
   #data_dir="/projects/katefgroup/xian/data",
   data_dir="/projects/katefgroup/fish/dynamics",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=32,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   #DEBUG_UNPROJECT=True,
   BS=2, valp=500, is_trainval_diff_summ=True,
   #load_name = "0517_bulletpush3D_4_multicam_bn_mask/no_bn_on_top_3d_shuffle"
   #load_name = "0517_bulletpush3D_4_multicam_bn_mask/xian_bs4"
   )


#####################3##### view prediction baseline #############################
# for cotrain only
CONVLSTM_DIM = None
CONVLSTM_STEPS = None
GQN3D_CONVLSTM_STOCHASTIC = False



OG("0520_bulletpush3D_4_multicam_bn_mask_nview1_vp",
   "bulletpushdata_mitpush_same_200",
   #"bulletpushdata_cotrain",
   #"bulletpushdata_mitpush_same_200_split",
   #IS_DUMP_VIS=False,
   MASK_AGENT=True,
   #OUTLINE_PRECOMPUTED = True,
   #USE_OUTLINE=True,
   #IS_NOT_BN_IN_2D = True,
   NUM_VIEWS=1,
   train_vp=True,
   train_on_val = True,
   # run_full=True,
   #IS_NOT_BN_IN_3D = True,
   lr=0.0001,
   opname="bulletpush3D_4_cotrain", H =64, W=64,
   #data_dir="/projects/katefgroup/xian/data",
   data_dir="/projects/katefgroup/fish/dynamics",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=32,
   max_T=0,
   CONVLSTM_DIM = 256,
   CONVLSTM_STEPS = 6,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   #DEBUG_UNPROJECT=True,
   BS=4, valp=500, is_trainval_diff_summ=True,
   #load_name = "0517_bulletpush3D_4_multicam_bn_mask/no_bn_on_top_3d_shuffle"
   #load_name = "0517_bulletpush3D_4_multicam_bn_mask/xian_bs4"

)

OG("rl_new",
   "mujocopickplace",
   #"bulletpushdata_cotrain",
   #"bulletpushdata_mitpush_same_200_split",
   #IS_DUMP_VIS=False,
   MASK_AGENT=False, 
   eager=False,
   #OUTLINE_PRECOMPUTED = True,
   #USE_OUTLINE=True,
   #IS_NOT_BN_IN_2D = True,
   IS_NOT_BN_IN_3D = True,
   lr=0.001,
   max_T = 0,
   opname="bulletpush3D_4_cotrain", H=64, W=64,
   CONVLSTM_STEPS = 6,
   #data_dir="/projects/katefgroup/xian/data",
   #data_dir="/home/mprabhud/rl/softlearning/softlearning/map3D",
   #data_dir="/media/shared/Documents/Research/VMGE/3d_temp/",
   data_dir="/home/adhaig/softlearning/softlearning/map3D",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=32,
   split_format=True,
   # max_T=1,
   CONVLSTM_DIM = 256,
   IS_VIEW_PRED=True,
   radius=0.4, boundary_to_center=0.2, fov=45, fs_2D=8, # 0.8 0.2 20
   DEBUG_UNPROJECT=False,
   BS=16, valp=500, is_trainval_diff_summ=True,
   run_full=False,
#    #ckpt_cfg_dir="/home/mprabhud/rl/softlearning/softlearning/map3D/ckpt_cfg",
#    ckpt_cfg_dir="/media/shared/Documents/Research/VMGE/3d_temp/ckpt_cfg",
#    load_name="rl_new/1"
   ckpt_cfg_dir="ckpt_cfg",
   load_name="rl_new/1",
   #ckpt_base = "/home/mprabhud/rl/softlearning/softlearning/map3D/"
   ckpt_base = "/media/shared/Research/VMGE/softlearning/softlearning/map3D/"
   #load_name = "0517_bulletpush3D_4_multicam_bn_mask/no_bn_on_top_3d_shuffle"
   #load_name = "0517_bulletpush3D_4_multicam_bn_mask/xian_bs4"
   )
OG("rl_temp","rl_new",ckpt_cfg_dir="ckpt_cfg")

OG("rl_new_reach","rl_new",load_name="rl_new/1")
OG("rl_new_reach_detect","rl_new",load_name="rl_new_detector/1",detector=True)


############################## SLAM Baseline #######################################

OG("0520_bulletpush3D_4_multicam_bn_mask_slam_view1",
   "bulletpushdata_0520_val_5steps_20",
   #"bulletpushdata_mitpush_same_200",
   #"bulletpushdata_cotrain",
   #"bulletpushdata_mitpush_same_200_split",
   #IS_DUMP_VIS=False,
   mode="test",
   MASK_AGENT=True,
   NUM_VIEWS=3,
   #eager=True,
   #OUTLINE_PRECOMPUTED = True,
   #USE_OUTLINE=True,
   #IS_NOT_BN_IN_2D = True,
   IS_NOT_BN_IN_3D = True,
   lr=0.001,
   opname="bulletpush3D_4_slam", H =64, W=64,
   #data_dir="/projects/katefgroup/xian/data",
   data_dir="/projects/katefgroup/fish/dynamics",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=32,
   max_T=5,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=16,
   #DEBUG_UNPROJECT=True,
   BS=1, valp=500, is_trainval_diff_summ=True,
   load_name = "0520_bulletpush3D_4_multicam_bn_mask_slam_view1/lr3_view3_no_bo_top_3d_maskagent"
   #load_name = "0517_bulletpush3D_4_multicam_bn_mask/xian_bs4"
   )















OG("0517_bulletpush3D_4_multicam_bn_debug",
   "bulletpushdata_mitpush_same_200",
   eager=True,
   #"bulletpushdata_basic_ht",
   #DEBUG_LOSSES=False,
   #IS_DUMP_VIS=False,
   #PRETRAIN_CONTACT=True,
   #IS_PREDICT_CONTACT=True,
   MASK_AGENT=True,
   #OUTLINE_PRECOMPUTED = True,
   #USE_OUTLINE=True,
   lr=0.001,
   opname="bulletpush3D_4", H =64, W=64,
   data_dir="/projects/katefgroup/fish/dynamics/",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=32,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   DEBUG_UNPROJECT=True,
   BS=2, valp=500, is_trainval_diff_summ=True,
   #load_name = "bulletpush3D_4_multicam/bs8_contact")
   )

OG("bulletpush3D_4_multicam_op",
   "bulletpushdata_mitpush_same_200",
   #eager=True,
   mode="train",
   BBOX_RANDOMIZATION=True,
   PRETRAIN_CONTACT=True,
   IS_PREDICT_CONTACT=True,
   OUTLINE_PRECOMPUTED = True,
   USE_OUTLINE=False,
   lr=0.001,
   opname="bulletpush3D_4", H =64, W=64,
   data_dir="/projects/katefgroup/fish/dynamics/",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=32,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   BS=8, valp=500, is_trainval_diff_summ=True)

OG("bulletpush3D_4_multicam_flow",
   "bulletpushdata_mitpush_same_200",
   #eager=True,
   #mode="test",
   #IS_DUMP_VIS=False,
   #BBOX_RANDOMIZATION=True,
   #PRETRAIN_CONTACT=True,
   #IS_PREDICT_CONTACT=True,
   #:OUTLINE_PRECOMPUTED = True,
   #IS_NOT_BN_IN_FLOW=False,
   #USE_3D_REFINE=False,
   MASK_AGENT=True,
   lr=0.001,
   opname="bulletpush3D_flow", H =64, W=64,
   data_dir="/projects/katefgroup/fish/dynamics/",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=16,
   max_T=1,
   #DEBUG_UNPROJECT=True,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   BS=2, valp=500, is_trainval_diff_summ=True,
   #load_name="bulletpush3D_4_multicam_flow/test"
)

OG("bulletpush3D_4_multicam_crop_flow",
   "bulletpushdata_mitpush_same_200",
   #eager=True,
   #IS_NOT_BN_IN_FLOW=False,
   #USE_3D_REFINE=False,
   MASK_AGENT=True,
   lr=0.001,
   opname="bulletpush3D_flow", H =64, W=64,
   data_dir="/projects/katefgroup/fish/dynamics/",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=16,
   eager=True,
   max_T=1,
   #DEBUG_UNPROJECT=True,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   BS=2, valp=500, is_trainval_diff_summ=True,
   #load_name="bulletpush3D_4_multicam_crop_flow/test_flow_action_everywhere_lr3"
)


OG("bulletpush3D_4_multicam_crop_flow_debug",
   "bulletpushdata_mitpush_same_200_split_debug1",
   #"bulletpushdata_mitpush_same_200",
   #eager=True,
   #mode="test",
   #IS_NOT_BN_IN_FLOW=False,
   #USE_3D_REFINE=False,
   MASK_AGENT=True,
   lr=0.001,
   opname="bulletpush3D_flow", H =64, W=64,
   data_dir="/projects/katefgroup/fish/dynamics/",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=16,
   max_T=1,
   #DEBUG_UNPROJECT=True,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   BS=1, valp=10, savep=200, is_trainval_diff_summ=True,
   #load_name="bulletpush3D_4_multicam_crop_flow_debug/test_tang"
)



OG("bulletpush3D_4_multicam_flow_test",
   "bulletpushdata_mitpush_same_200",
   #eager=True,
   mode="test",
   #IS_DUMP_VIS=False,
   #BBOX_RANDOMIZATION=True,
   #PRETRAIN_CONTACT=True,
   #IS_PREDICT_CONTACT=True,
   #:OUTLINE_PRECOMPUTED = True,
   IS_NOT_BN_IN_FLOW=False,
   lr=0.1,
   opname="bulletpush3D_flow", H =64, W=64,
   data_dir="/projects/katefgroup/fish/dynamics/",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=32,
   max_T=1,
   #DEBUG_UNPROJECT=True,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=4,
   BS=3, valp=500, is_trainval_diff_summ=True,
   load_name="bulletpush3D_4_multicam_flow/test"
)



OG("bulletpush3D_4_multicam_test",
   "bulletpushdata_mitpush_same_200",
   #"bulletpushdata_basic_ht",
   mode="test",
   #eager=True,
   DEBUG_LOSSES=False,
   IS_DUMP_VIS=False,
   PRETRAIN_CONTACT=True,
   IS_PREDICT_CONTACT=True,
   lr=0.001,
   opname="bulletpush3D_4", H =64, W=64,
   data_dir="/projects/katefgroup/fish/dynamics/",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=16,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   #DEBUG_UNPROJECT=True,
   BS=8, valp=500, is_trainval_diff_summ=True,
   load_name = "bulletpush3D_4_multicam/bs8_contact"
   )

OG("bulletpush3D_4_multicam_no_contact_test",
   "bulletpushdata_mitpush_same_200",
   #"bulletpushdata_basic_ht",
   mode="test",
   #eager=True,
   DEBUG_LOSSES=False,
   IS_DUMP_VIS=False,
   lr=0.001,
   opname="bulletpush3D_4", H =64, W=64,
   data_dir="/projects/katefgroup/fish/dynamics/",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=16,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=16,
   #DEBUG_UNPROJECT=True,
   BS=9, valp=500, is_trainval_diff_summ=True,
   load_name = "bulletpush3D_4_multicam/0512_fs16_b8"
   )


OG("bulletpush_multicam",
   "bulletpushdata_mitpush_same_200",
   #"bulletpushdata_basic_h2",
   lr=0.01,
   ARCH="convfc",
   #eager=True,
   mode="test",
   BS=1,
   opname="bulletpush", H = 128, W=128,
   data_dir="/projects/katefgroup/fish/dynamics/",
   #eager=True,
   USE_ORN_IN_FIRST_STEP=True,
   boundary_to_center=1.5, valp=1000, is_trainval_diff_summ=True,
   load_name = "bulletpush_multicam/0511_push_multicam_lr2") #, lr=1E-8)

OG("bulletpush_multicam_test",
   "bulletpushdata_mitpush_same_200",
   #"bulletpushdata_basic_ht",
   mode="test",
   #eager=True,
   DEBUG_LOSSES=False,
   IS_DUMP_VIS=False,
   lr=0.01,
   ARCH="convfc",
   #mode="test",
   #eager=True,
   BS=12,
   opname="bulletpush", H = 128, W=128, data_dir="/projects/katefgroup/fish/dynamics/",
   #eager=True,
   USE_ORN_IN_FIRST_STEP=True,
   boundary_to_center=1.5, valp=1000, is_trainval_diff_summ=True,
   load_name = "bulletpush_multicam/lr3_angle_l1_loss") #, lr=1E-8)


OG("bulletpush3D_2d_multicam",
   "bulletpushdata_mitpush_same_200",
   #"bulletpushdata_basic_h2", #_airplane",
   #eager=True,
   #mode="test",
   lr = 0.001,
   opname="bulletpush3D_2d",
   H =64, W=64, data_dir="/projects/katefgroup/fish/dynamics",
   AGGREGATION_METHOD = 'gru', USE_MESHGRID=False,
   crop_size=32,
   crop_feat_size=0.125,
   #eager=True,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   DEBUG_UNPROJECT=False, BS=8, valp=500, is_trainval_diff_summ=True)


OG("bulletpush3D_2d_multicam_test",
   "bulletpushdata_mitpush_same_200",
   mode="test",
   DEBUG_LOSSES=False,
   IS_DUMP_VIS=False,
   lr = 0.001,
   opname="bulletpush3D_2d",
   H =64, W=64, data_dir="/projects/katefgroup/fish/dynamics",
   AGGREGATION_METHOD = 'gru', USE_MESHGRID=False,
   crop_size=32,
   crop_feat_size=0.125,
   #eager=True,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   DEBUG_UNPROJECT=False, BS=12, valp=500, is_trainval_diff_summ=True,
   load_name="bulletpush3D_2d_multicam/lr2_bs8_with_cam")

OG("bulletpush3D_4",
   "bulletpushdata_basic_h2",
   #"bulletpushdata_basic_ht",
   #mode="test",
   #eager=True,
   #PRETRAIN_CONTACT=True,
   lr=0.0001,
   opname="bulletpush3D_4", H =64, W=64,
   data_dir="/projects/katefgroup/fish/dynamics/",
   AGGREGATION_METHOD = 'average', USE_MESHGRID=False,
   crop_size=16,
   max_T=1,
   radius = 5.0, boundary_to_center=1.5, fov=36, fs_2D=8,
   DEBUG_UNPROJECT=False,
   #IS_PREDICT_CONTACT=True,
   BS=12, valp=500, is_trainval_diff_summ=True,
   load_name="bulletpush3D_4/0509_lr4_h2"
)



OG('doublemug',
   'doubledata',
   MULTI_UNPROJ = True, lr = 5E-4, BS = 2, valp = 1, H = 128, W = 128, data_dir = "double_tfrs/",
   MINV = 20, MAXV = 20, VDELTA = 20
)

OG('doublemug_debug',
   'doublemug', 'doubledebugdata',
   DEBUG_VOXPROJ = True
)

OG('doublemug_train',
   'doublemug',
   valp = 100, BS = 2
)

OG('doublemug_small',
   'doublemug_train',
   S = 64,
)

OG('doublemug_small_debug',
   'doublemug_small', 'doubledebugdata',
   DEBUG_VOXPROJ = False, DEBUG_UNPROJECT = True, valp = 50, BS = 4,
)

#what is voxproj vs unproj?
# works fine for
# no debug voxproj (single data) +
# debug voxproj? 
# no debug voxproj (all data) +
# debug unproj ??? seems fishy -- not sure if it works
OG('doublemug_small2_debug',
   'doublemug_small_debug',
   S = 32, H = 64, W = 64,
)

OG('doublemug_train_gru',
   'doublemug_train',
   AGGREGATION_METHOD = 'gru', BS = 1
)

#works w/ depth/mask
#works w/o depth/mask
OG('querytask',
   'doublemug_train_gru',
   opname = 'query',
   RANDOMIZE_BG = False, AGGREGATION_METHOD = 'average', BS = 2, lr = 1E-4,
   USE_OUTLINE = False, USE_MESHGRID = False, AUX_POSE = False
)

OG('querytask_debug',
   'querytask', 'doubledebugdata',
   lr = 1E-4
)

OG('size64',
   H = 64, W = 64, BS = 8,
)

#works
OG('querytask_debug64',
   'querytask_debug', 'size64'
)

#not sure if works w/ depth/mask
#doesn't work w/o depth/mask
#wait... this works!
OG('querytask64',
   'querytask', 'size64'
)

OG('querytask_eager',
   'querytask_debug', 
   eager = True, BS = 1, NUM_VIEWS = 2
)

OG('ego',
   'querytask',
   opname="ego", fs_2D=16, NUM_PREDS = 1, H = 64, W = 64, S = None, BS=1,
   USE_OUTLINE=True
)

OG('ego_gru',
   'querytask',
   opname="ego", fs_2D=16, NUM_PREDS = 1, H = 64, W = 64, S = None, BS=1,
   AGGREGATION_METHOD = 'gru', USE_OUTLINE=True)

OG('ego_rpn',
   'querytask',
   opname="ego_rpn", fs_2D=8, NUM_PREDS = 1, H = 64, W = 64, S = None, BS=2)
#USE_OUTLINE=True)

"""
OG('ego_rpn_gru',
   'querytask', 'double15data',
   savep=2000,
   #load_name="ego_rpn_gru/debug",
   opname="ego_rpn", fs_2D=16, NUM_PREDS = 1, H = 64, W = 64, S = None, BS=1,
   AGGREGATION_METHOD = 'gru') #, USE_OUTLINE=True)
OG('ego_rpn_gru',
   'querytask', 'double15data',
   opname="ego_rpn", fs_2D=16, NUM_PREDS = 1, H = 64, W = 64, S = None, BS=1,
   AGGREGATION_METHOD = 'gru') #, USE_OUTLINE=True)
"""
OG('ego_rpn_gru_ego_pretrain',
   'querytask', #'double110data',
   savep=1000,
   USE_PREDICTED_EGO=True,
   lr = 1E-2,
   EGO_PRETRAIN=True,
   #load_name="ego_rpn_gru_depth_pretrain/tanh_l1_lr3",
   opname="ego_rpn", fs_2D=8, NUM_PREDS = 1, H = 64, W = 64, S = None, BS=2,
    AGGREGATION_METHOD = 'gru', USE_OUTLINE=True)

OG('ego_rpn_gru_depth_pretrain',
   'querytask', #'double110data',
   savep=1000,
   lr = 1E-3,
   USE_PREDICTED_DEPTH=True,
   DEPTH_PRETRAIN=True,
   load_name="ego_rpn_gru_depth_pretrain/tanh_l1_lr3",
   opname="ego_rpn", fs_2D=8, NUM_PREDS = 1, H = 64, W = 64, S = None, BS=2,
    AGGREGATION_METHOD = 'gru', USE_OUTLINE=True)

OG('ego_rpn_gru_rpn_pretrain',
   'querytask', #'double110data',
   #lr = 1E-3,
   USE_PREDICTED_EGO=True, #False, #True,
   savep=1000,
   #USE_PREDICTED_DEPTH=True,
   #load_name = "ego_rpn_gru_depth_pretrain/tanh_l1_lr3",
   #load_name = "ego_rpn_gru_rpn_pretrain/from_depth_pretrain_l1_tanh_conti_lr3",
   #load_name = "ego_rpn_gru_rpn_pretrain/0401_pretrain",
   load_name = "ego_rpn_gru_ego_pretrain/ego_pretrain_one_layer",
   opname="ego_rpn", fs_2D=8, NUM_PREDS = 1, H = 64, W = 64, S = None, BS=2,
    AGGREGATION_METHOD = 'gru', USE_OUTLINE=False)

OG('ego_rpn_gru_debug5',
   'querytask', #'double110data',
   mode="test",
   lr = 1E-3,
   #mode="test",
   #USE_PREDICTED_DEPTH=True,
   #USE_PREDICTED_EGO=True, #False, #True,
   savep=1000,
   #mode="test",
   #FIX_VIEW=True,
   #load_name="ego_rpn_gru/pretrain_910data_fs16_bs1_cam_only",
   #load_name="ego_rpn_gru/maskrcnn_all_rpn_only",
   #load_name="ego_rpn_gru/ego_rpn_gru_cam_only_1116",
   #load_name="ego_rpn_gru/maskrcnn_all_0_01_c_conti",
   #load_name = "ego_rpn_gru_debug/all_outline_overfit",
   #load_name = "ego_rpn_gru_debug/all_outline_overfit_1scene",
   #load_name = "ego_rpn_gru_debug3/predict_bbox_large_bbox_loss",
   #load_name = "ego_rpn_gru_debug4/3d_outline_pretrain",
   #load_name = "ego_rpn_gru_rpn_pretrain/3d_gru_pretrain_with_depth_rpn_3d_pretrain_depth",
   #load_name = "ego_rpn_gru_rpn_pretrain/3d_gru_pretrain_with_depth_test",
   #load_name = "ego_rpn_gru_debug5/learnt_depth_with_depth_pretrain_test",
   #load_name = "ego_rpn_gru_rpn_pretrain/0401_pretrain",
   #load_name = "ego_rpn_gru_debug5/from_0402_stage1_anchor_size_025_29000",
   #load_name = "ego_rpn_gru_rpn_pretrain/0403_predict_depth",
   #load_name = "ego_rpn_gru_debug5/0403_predict_ego",
   #load_name="ego_rpn_gru_rpn_pretrain/from_depth_pretrain_l1_tanh_conti_lr3",
   #:load_name = "ego_rpn_gru_rpn_pretrain/0402_stage1_pretrain",
   #load_name = "ego_rpn_gru_rpn_pretrain/0402_stage1_anchor_size_025_29000",
   #load_name = "ego_rpn_gru_ego_pretrain/ego_pretrain_one_layer", 
   #load_name = "ego_rpn_gru_debug5/0403_predict_ego_lr2_cam_loss_01",
   #load_name = "ego_rpn_gru_debug5/from_ego_pretrain_one_layer",
   load_name="ego_rpn_gru_debug5/gt_depth",#from_depth_pretrain_l1_tanh_conti_lr3",
   #USE_GT_ROIS = True,
   #lr = 1E-10,
   opname="ego_rpn", fs_2D=8, NUM_PREDS = 1, H = 64, W = 64, S = None, BS=2,
    AGGREGATION_METHOD = 'gru', USE_OUTLINE=True) #, DEBUG_UNPROJECT=True)

OG('ego_rpn_mat',
   'xian_doubledata',
   eager=True,
   USE_PREDICTED_EGO=False, #True,
   savep=1000,
   DEBUG_UNPROJECT=True,
   boundary_to_center=2.5,
   fov=60,
   data_dir="/projects/katefgroup/datasets/blender_flex_cam_tfrs",
   opname="ego_rpn_mat", fs_2D=8, NUM_PREDS = 1, H = 128, W = 128, S = None, BS=2,
   MINV = 20, VDELTA = 20,
   AGGREGATION_METHOD = 'gru', USE_OUTLINE=True)

OG('ego_rpn_mat_realsense',
   'realsense_data',
   eager=True,
   USE_PREDICTED_EGO=False, #True,
   savep=1000,
   DEBUG_UNPROJECT=True,
   boundary_to_center=0.2,
   radius=0.2,
   data_dir="/projects/katefgroup/fish/realsense",
   opname="ego_rpn_mat", fs_2D=8, NUM_PREDS = 1, H = 128, W = 128, S = None, BS=2,
   MINV = 20, VDELTA = 20,
   AGGREGATION_METHOD = 'gru', USE_OUTLINE=True)




OG('ego_rpn_2d_debug4',
   'querytask', #'double130data',
   mode="exp_rpn",
   savep = 2000,
   load_name = "ego_rpn_2d_debug4/0403_with_mask_branch",
   opname="ego_rpn", net_name="2d", fs_2D=16, NUM_PREDS = 1, H = 64, W = 64, S = None, BS=1, AGGREGATION_METHOD='gru', USE_OUTLINE=True)
#, opname="ego", NUM_PREDS = 1, H = 64, W = 64, S = None, BS=1
#)

OG('ego_rpn_2d_gru',
   'querytask',
   savep=2000,
   #load_name="ego_rpn_2d_gru/rpn_2d_gru_1116",
   #load_name = "ego_rpn_2d_gru/ego_rpn_cam_only_1116",

   opname="ego_rpn", net_name="2d", fs_2D=32, NUM_PREDS = 1, H = 64, W = 64, S = None, BS=2,
   AGGREGATION_METHOD = 'gru')

OG('gqnbase',
   'querytask',
   NUM_PREDS = 1, H = 64, W = 64, S = None
)

OG('gqn2d',
   'gqnbase',
   opname = 'gqn2d', BS = 32
)

OG('gqn3d',
   'gqnbase',
   opname = 'gqn3d', BS = 8
)

#converges to 0 quickly
OG('gqn3d_convlstm',
   'gqn3d',
   GQN3D_CONVLSTM = True, valp = 10,
)

OG('gqn3dv2', 'gqn3d', lr = 5E-4)
OG('gqn3dv3', 'gqn3d', lr = 2E-5)


#works :)
OG('gqn3d_debug',
   'gqn3d', 'doubledebugdata',
)

OG('mnist',
   opname = 'mnist', BS = 64, valp = 200,
)

OG('mnist_convlstm',
   'mnist',
   MNIST_CONVLSTM = True,
)

OG('mnist_convlstm_stoch',
   'mnist_convlstm',
   MNIST_CONVLSTM_STOCHASTIC = True,
)


def set_experiment(exp_name):
    print('running experiment', exp_name)
    for key, value in options.get(exp_name).items():
        _verify_(key, value)
        globals()[key] = value 

def _verify_(key, value):
    print(key, '<-', value)
    assert key in globals(), ('%s is new variable' % key)

if exp_name not in options._options_:
    print('*' * 10 + ' WARNING -- no option group active ' + '*' * 10)
else:
    print('running experiment', exp_name)
    for key, value in options.get(exp_name).items():
        _verify_(key, value)
        globals()[key] = value

#stuff which must be computed afterwards
        
# camera stuffs
fx = W / 2.0 * 1.0 / math.tan(fov * math.pi / 180 / 2)
fy = fx
focal_length = fx / (W / 2.0) #NO SCALING

x0 = W / 2.0
y0 = H / 2.0
