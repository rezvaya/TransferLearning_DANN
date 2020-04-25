import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

def train (model, optimizer, train_loader, test_loader,train_ds, test_dataset, num_epochs, best_model_path):
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model
    model.train()
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)            
            _, preds = torch.max(outputs, 1)

            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / float(len(train_ds))
            epoch_acc = running_corrects.double() / len(train_ds)
                       
        print('____________Loss: {:.4f} train_acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        test_accuracy=test(test_loader, test_dataset, model)
        
        if test_accuracy > best_accuracy:
            torch.save(model.state_dict(), best_model_path)
            best_accuracy = test_accuracy

    return model
            
        
def test(test_loader,test_dataset, model):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    test_correct_count = 0.0   
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)   

        _, preds = torch.max(outputs, 1)       
        test_correct_count += torch.sum(preds == labels.data)

    epoch_acc =  test_correct_count.double() / len(test_dataset)
    print('_______Test_acc %f' % (epoch_acc))
        
    return epoch_acc