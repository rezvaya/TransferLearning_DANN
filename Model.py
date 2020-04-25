import torchvision
from torchvision import datasets, models, transforms


def get_classifier(feature_extractor, num_classes):
    clf = nn.Sequential()
    clf.add_module('c_fc1', nn.Linear(feature_extractor.last_channel,32))
    clf.add_module('c_relu1', nn.ReLU(True))
    clf.add_module('c_bn1', nn.BatchNorm1d(32))
    clf.add_module('c_fc2', nn.Linear(32, 32))
    clf.add_module('c_relu2', nn.ReLU(True))
    clf.add_module('c_bn2', nn.BatchNorm1d(32))
    clf.add_module('c_fc3', nn.Linear(32, num_classes))
    clf.add_module('c_softmax', nn.LogSoftmax(dim = 1))
    return clf

def get_linear_classifier(input_dim, num_classes):
    clf = nn.Sequential(
        nn.Linear(input_dim,32),
        nn.LogSoftmax(dim = 1),
    )
    return clf

def get_mobilenet_model(num_freeze_layers, num_classes):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transfer_net = models.mobilenet_v2(pretrained = True)

    ct = 0

    for layer in transfer_net.children():
        ct = ct + 1
        if ct < num_freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

    transfer_net.classifier = get_classifier(transfer_net, num_classes)
    transfer_net = transfer_net.to(device)

    return transfer_net


def get_resnet_model(num_freeze_layers, num_classes):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transfer_net = models.resnet34(pretrained = True)

    ct = 0

    for layer in transfer_net.children():
        ct = ct + 1
        if ct < num_freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

    transfer_net.classifier = get_linear_classifier(transfer_net.fc.in_features, num_classes)
    transfer_net = transfer_net.to(device)

    return transfer_net

    