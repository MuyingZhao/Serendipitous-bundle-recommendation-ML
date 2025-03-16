import subprocess

subprocess.call(['pip', 'install', "tensorboardX"])

subprocess.call(['pip', 'install', "setproctitle"])

import os

try:
    os.chdir("drive/MyDrive/recommendation/BGCN")
except:
    pass
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
import setproctitle
import dataset
from model import BGCN, BGCN_Info
from utils import check_overfitting, early_stop, logger
from train import train
from metric import Recall, NDCG, MRR
from config import CONFIG
from test1 import test
import loss
from itertools import product
import time
from tensorboardX import SummaryWriter


TAG = ''

model = torch.load("model_file_from_simple_sample_model.pth")
model.eval()

test_user = torch.tensor([[10675]]) # this is the user id
test_bundles = torch.tensor([[312,312]]) # this is the bundle id 

# this is the forward function, that given input giving output.
model.forward(test_user, test_bundles) 
# this idea is using the user/bundle id to get user/bundle embedding and do the calculation, and then feed to model

# another way, but basically the same  
users_feature, bundles_feature = model.propagate()

test_users_embedding = [i[test_user].expand(- 1, test_bundles.shape[1], -1) for i in users_feature]
test_bundles_embedding = [i[test_bundles] for i in bundles_feature] 
model.predict(test_users_embedding, test_bundles_embedding)