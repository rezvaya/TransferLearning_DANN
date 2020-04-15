import numpy as np
import os

from matplotlib import pyplot as plt

from Train_test import train, test
from Model import get_model, get_classifier
from Data_prepare import prepare_datasets
from Data_prepare import AllCropsDataset

DATA_PATH_SOURCE='color'
DATA_PATH_TARGET='pdd'
BATCH_SIZE=64
NUM_FREEZE_LAY=12
BEST_SOURCE_MODEL_PATH = 'best_model_10FL.pth'
lr=0.0003
NUM_EPOCH=100

###load source data
train_source_ds, test_source_ds = prepare_datasets(DATA_PATH_SOURCE) 

train_source_loader = torch.utils.data.DataLoader(train_source_ds, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)
test_source_loader = torch.utils.data.DataLoader(test_source_ds, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)

NUM_SORCE_CLASSES=len(train_ds.classes)

###load target data
train_target_ds, test_target_ds = prepare_datasets(DATA_PATH_TARGET) 

train_target_loader = torch.utils.data.DataLoader(train_target_ds, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)
test_target_loader = torch.utils.data.DataLoader(test_target_ds, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)

###train net on source data
transfer_net=get_model(NUM_FREEZE_LAY, NUM_SORCE_CLASSES, lr)
optimizer =  optim.Adam(transfer_net.parameters(), lr=lr, eps=1e-08)

transfer_net=train(transfer_net, optimizer, train_source_loader, test_source_loader, test_source_ds, NUM_EPOCH,BEST_MODEL_PATH)


