import torchvision
from torchvision import datasets, models, transforms


def get_classifier(feature_extractor, NUM_CLASSES):
    clf = nn.Sequential()
    clf.add_module('c_fc1', nn.Linear(feature_extractor.last_channel,32))
    clf.add_module('c_bn1', nn.BatchNorm1d(32))
    clf.add_module('c_relu1', nn.ReLU(True))
    clf.add_module('c_drop1', nn.Dropout2d())
    clf.add_module('c_fc2', nn.Linear(32, 32))
    clf.add_module('c_bn2', nn.BatchNorm1d(32))
    clf.add_module('c_relu2', nn.ReLU(True))
    clf.add_module('c_fc3', nn.Linear(32, NUM_CLASSES))
    clf.add_module('c_softmax', nn.LogSoftmax(dim=1))
    return clf

def get_model(NUM_FREEZE_LAY, NUM_CLASSES, lr):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transfer_net = models.mobilenet_v2(pretrained=True)

    ct = 0

    for layer in transfer_net.children():
        ct = ct + 1
        if ct < NUM_FREEZE_LAY:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

    #classifier=get_classifier(transfer_net, NUM_CLASSES)
    classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_extractor.last_channel, NUM_CLASSES),
        )
        
    transfer_net.classifier=classifier    

    transfer_net = transfer_net.to(device)

    return transfer_net