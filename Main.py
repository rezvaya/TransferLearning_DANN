import numpy as np
import os

from matplotlib import pyplot as plt

from Train_and_test import train, test
from Model import get_model, get_classifier
from Data_prepare import prepare_datasets
from Data_prepare import AllCropsDataset

data_path_source = 'color'
data_path_target = 'pdd'
batch_size = 64
num_freeze_layers = 12
best_source_model_path = 'best_model_source_train.pth'
best_target_model_path = "best_model_target_train.pth"
lr_source_net = 0.00003
lr_target_net = 0.000003
num_epoch_st = 500
num_epoch_tt = 100

###load source data
train_source_ds, test_source_ds = prepare_datasets(data_path_source) 

train_source_loader = torch.utils.data.DataLoader(
    train_source_ds,
     pin_memory = True,
     batch_size = batch_size,
     shuffle = True,
     num_workers = 16
     )
test_source_loader = torch.utils.data.DataLoader(
    test_source_ds,
    pin_memory = True,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 16
    )

num_source_classes = len(train_source_ds.classes)

###load target data
train_target_ds, test_target_ds = prepare_datasets(data_path_target) 

train_target_loader = torch.utils.data.DataLoader(
    train_target_ds,
    pin_memory = True,
    batch_size = batch_size, 
    shuffle = True, 
    num_workers = 16
    )
test_target_loader = torch.utils.data.DataLoader(
    test_target_ds, 
    pin_memory = True, 
    batch_size = batch_size, 
    shuffle = True, 
    num_workers = 16
    )

num_target_classes = len(train_target_ds.classes)

###train net on source data
transfer_net = get_resnet_model(num_freeze_layers, num_source_classes)
optimizer = optim.Adam(transfer_net.parameters(), lr = lr_source_net, eps = 1e-08)

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "cpu"
    )

transfer_net.to(device)

transfer_net = train(
    transfer_net, 
    optimizer, 
    train_source_loader, 
    test_source_loader, 
    train_source_ds, 
    test_source_ds, 
    num_epoch_st, 
    best_source_model_path
    )

###load pretrained on source data model
pretrained_model = get_resnet_model(num_freeze_layers, num_sorce_classes)
pretrained_model.load_state_dict(torch.load(best_source_model_path))
pretrained_model.eval()

###test on source data pretrained model
test_accuracy_by_load_model = test(test_source_loader, test_source_ds, pretrained_model)

###change the classifier
pretrained_model.classifier = get_linear_classifier(pretrained_model.fc.in_features, num_target_classes)

###train model on target dataset
optimizer = optim.Adam(pretrained_model.parameters(), lr = lr_target_net, eps = 1e-08)
pretrained_model = train(
    pretrained_model, 
    optimizer, 
    train_target_loader, 
    test_target_loader, 
    train_target_ds, 
    test_target_ds, 
    num_epoch_tt, 
    best_target_model_path
    )

###test on target data pretrained model
target_test_accuracy = test(test_target_loader, test_target_ds, pretrained_model)

