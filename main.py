# 调用官方库及第三方库
import torch
from torch.utils.data import DataLoader
import numpy as np
import datetime
import platform
import argparse
import random
import os

# 基础功能
from common.cdloader import CDReader, TestReader
from work.train import train
from work.predict import predict

# from changedetection.configs.config import get_config
# from changedetection.models.MambaBCD import STMambaBCD
# from models.model import ChangeACFM, ChangeMM,ChangeResMM
# from models.model2 import ChangeResSR, ChangeVitSR, ChangeSR, ChangeSR_noMCF
from lccdmamba.model import LCCDMamba, LCCDMamba_noDTMS, LCCDMamba_RM, LCCDMamba_MISFPara, LCCDMamba_Lite, LCCDMamba_Lite2
from rsmamba import RSMamba_CD
from common.ready import Args


# dataset_name = "GVLM_CD"
# dataset_name = "LEVIR_CD"
# dataset_name = "CLCD"
# dataset_name = "WHU_BCD"

# dataset_name = "MacaoCD"
# dataset_name = "SYSU_CD"
dataset_name = "S2Looking"

dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)

num_classes = 2
batch_size = 2
num_epochs = 100 

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

# config = get_config(mparas)

model = RSMamba_CD()


model_name = model.__str__().split("(")[0]
args = Args('output/{}'.format(dataset_name.lower()), model_name)
args.data_name = dataset_name
args.num_classes = num_classes
args.batch_size = batch_size
args.iters = num_epochs
args.pred_idx = 0
args.device = 1

def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    # 代码运行预处理
    seed_torch(32765)
    torch.cuda.empty_cache()
    torch.cuda.init()
    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(args.device)
    device = torch.device(args.device)
    
    eval_data = CDReader(path_root = dataset_path, mode="val", en_edge=False)
    train_data = CDReader(path_root = dataset_path, mode="train", en_edge=False)
    
    # dataloader_pred = DataLoader(pred_data, batch_size, num_workers=1)
    dataloader_eval = DataLoader(dataset=eval_data, batch_size=args.batch_size, num_workers=16,
                                 shuffle=False, drop_last=True)
    dataloader_train = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=16,
                                  shuffle=True, drop_last=True)
    
    test_data = TestReader(path_root = dataset_path, mode="test", en_edge=False)
    dataloader_test = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=0,
                                  shuffle=True, drop_last=True)
    
    # try:
    #     model.load_state_dict(torch.load(save_model_dir))
    #     print("load success")
    # except:
    #     args.num_epochs = 300
    #     args.params["lr"] = 0.0005
    model = model.to(device, dtype=torch.float)
    # model.load_state_dict(torch.load("/home/jq/Code/torch/output/levir_d/SiamUnet_diff_2023_10_26_16/SiamUnet_diff_best.pth"))
    train(model, dataloader_train, dataloader_eval, dataloader_test, args)
    # weight_path = r"/home/jq/Code/VMamba/output/whu_bcd/ChangeSR_2024_06_25_16/ChangeSR_best.pth"
    # predict(model,test_data,weight_path, dataset_name)
    
