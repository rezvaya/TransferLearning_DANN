import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

def train (model, optimizer, train_loader, test_loader, test_dataset, NUM_EPOCHS, BEST_MODEL_PATH):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model
    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        
        print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        i=0
        for images, labels in iter(train_loader):
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
            
           
            i+= 1

        print('____________Loss: {:.4f} train_acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        test_accuracy=test(test_loader, test_ds, model)
        
        if test_accuracy > best_accuracy:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_accuracy = test_accuracy

    return model

    
          
        
def test(test_loader,test_dataset, model):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_error_count = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
    
    test_accuracy = 1.0 - (float(test_error_count) / float(len(test_ds)))
    print('_______Test_acc %f' % (test_accuracy))

        
    return test_accuracy