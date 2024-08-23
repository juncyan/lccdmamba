import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict



import torch
import torch.nn as nn
import torch.nn.functional as F


from .decoder import Decoder
from .backbone import Backbone_VSSM #, Backbone_VSSM_samll


class LCCDMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3))
       
        self.decoder = Decoder(
            dims=self.encoder.dims,
            out_channels=64,
            channel_first=self.encoder.channel_first)

        self.main_clf = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)


    def forward(self, pre_data, post_data=None):
        if post_data == None:
            post_data = pre_data[:,3:,:,:]
            pre_data = pre_data[:,:3,:,:]
        
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        
        output = self.decoder(pre_features, post_features)

        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        output = self.main_clf(output)

        return output
    