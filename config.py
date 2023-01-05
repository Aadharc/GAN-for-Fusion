import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR_VIS = "data_2/vis/train"
VAL_DIR_VIS = "data_2/vis/val"
TRAIN_DIR_IR = "data_2/ir/train"
VAL_DIR_IR = "data_2/ir/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA = 100
# LAMBDA_GP = 10
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC_IR = "disc_ir_Two_Mask_moredata.pth.tar"
CHECKPOINT_DISC_VIS = "disc_vis_Two_Mask_moredata.pth.tar"
CHECKPOINT_GEN = "gen_10_Two_Mask_moredata.pth.tar"



