import numpy as np
import os

from matplotlib import pyplot as plt

from Train_and_test import train, test
from Model import get_model, get_classifier
from Data_prepare import prepare_datasets
from Data_prepare import AllCropsDataset

DATA_PATH_SOURCE='color'
DATA_PATH_TARGET='pdd'
BATCH_SIZE=64
NUM_FREEZE_LAY=12
BEST_SOURCE_MODEL_PATH = 'best_model_source_train.pth'
BEST_TARGET_MODEL_PATH="best_model_target_train.pth"
lr=0.00003
NUM_EPOCH_ST=500
NUM_EPOCH_TT=100

###load source data
train_source_ds, test_source_ds = prepare_datasets(DATA_PATH_SOURCE) 

train_source_loader = torch.utils.data.DataLoader(train_source_ds, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)
test_source_loader = torch.utils.data.DataLoader(test_source_ds, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)

NUM_SORCE_CLASSES=len(train_source_ds.classes)

###load target data
train_target_ds, test_target_ds = prepare_datasets(DATA_PATH_TARGET) 

train_target_loader = torch.utils.data.DataLoader(train_target_ds, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)
test_target_loader = torch.utils.data.DataLoader(test_target_ds, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)

NUM_TARGET_CLASSES=len(train_target_ds.classes)
###train net on source data
transfer_net=get_model(NUM_FREEZE_LAY, NUM_SORCE_CLASSES)
optimizer =  optim.Adam(transfer_net.parameters(), lr=lr, eps=1e-08)

transfer_net=train(transfer_net, optimizer, train_source_loader, test_source_loader, train_source_ds, test_source_ds, NUM_EPOCH_ST, BEST_SOURCE_MODEL_PATH)

###load pretrained on source data model
pretrained_model=get_model(NUM_FREEZE_LAY, NUM_SORCE_CLASSES)
pretrained_model.load_state_dict(torch.load(BEST_SOURCE_MODEL_PATH))
pretrained_model.eval()

###test on source data pretrained model
target_test_accuracy_by_load_model=test(test_source_loader, test_source_ds, pretrained_model)

###change the classifier
clf=get_classifier(pretrained_model, NUM_TARGET_CLASSES)
pretrained_model.classifier=clf

###train model on target dataset
optimizer = optim.Adam(pretrained_model.parameters(), lr=lr, eps=1e-08)
pretrained_model=train(pretrained_model, optimizer, train_target_loader, test_target_loader, train_target_ds, test_target_ds, NUM_EPOCH_TT, BEST_TARGET_MODEL_PATH)

