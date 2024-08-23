# 调用官方库及第三方库
import torch
from torch.utils.data import DataLoader
import numpy as np
import datetime
import platform
import random
import os
import argparse
from changedetection.configs.config import get_config
# 基础功能
from common.cdloader import TestReader
from work.predict import predict

from models.model2 import ChangeResSR, ChangeVitSR, ChangeSR, ChangeSR_noMCF, ChangeSR_Trans
from models.model import ChangeACFM, ChangeMM
from lccdmamba.model import LCCDMamba


parser = argparse.ArgumentParser(description="Training on SYSU/LEVIR-CD+/WHU-CD dataset")
parser.add_argument('--cfg', type=str, default='/home/jq/Code/VMamba/changedetection/configs/vssm1/vssm_base_224.yaml')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')

mparas = parser.parse_args()

    # with open(args.test_data_list_path, "r") as f:
    #     # data_name_list = f.read()
    #     test_data_name_list = [data_name.strip() for data_name in f]
    # args.test_data_name_list = test_data_name_list

config = get_config(mparas)
# model = ChangeSR(
#             pretrained="/home/jq/Code/weights/vssm_base_0229_ckpt_epoch_237.pth",
#             patch_size=config.MODEL.VSSM.PATCH_SIZE, 
#             in_chans=config.MODEL.VSSM.IN_CHANS, 
#             num_classes=config.MODEL.NUM_CLASSES, 
#             depths=config.MODEL.VSSM.DEPTHS, 
#             dims=config.MODEL.VSSM.EMBED_DIM, 
#             # ===================
#             ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
#             ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
#             ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
#             ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
#             ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
#             ssm_conv=config.MODEL.VSSM.SSM_CONV,
#             ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
#             ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
#             ssm_init=config.MODEL.VSSM.SSM_INIT,
#             forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
#             # ===================
#             mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
#             mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
#             mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
#             # ===================
#             drop_path_rate=config.MODEL.DROP_PATH_RATE,
#             patch_norm=config.MODEL.VSSM.PATCH_NORM,
#             norm_layer=config.MODEL.VSSM.NORM_LAYER,
#             downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
#             patchembed_version=config.MODEL.VSSM.PATCHEMBED,
#             gmlp=config.MODEL.VSSM.GMLP,
#             use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#             ) 
model = LCCDMamba()


dataset_name = "GVLM_CD"
# dataset_name = "LEVIR_CD"
# dataset_name = "CLCD"
# dataset_name = "SYSCD_d"
# dataset_name = "MacaoCD"
# dataset_name = "WHU_BCD"
# dataset_name = "S2Looking"

dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)



if __name__ == "__main__":
    # 代码运行预处理
    torch.cuda.empty_cache()
    torch.cuda.init()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    

    test_data = TestReader(dataset_path, mode="test",en_edge=False)
    
    weight_path = r"/home/jq/Code/VMamba/output/s2looking/LCCDMamba_2024_07_17_12/LCCDMamba_best.pth"
    predict(model, test_data, weight_path,test_data.data_name,2,0)
    # x = torch.rand([1,3,256,256]).cuda()
    # y = model(x,x)
    # print(y.shape)    
