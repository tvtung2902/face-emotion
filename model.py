import torch.nn as nn
import torchvision.models as models

def load_resnet(num_classes=7, pretrained=True, fine_tune=True):
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(layer in name for layer in ["layer2", "layer3", "layer4", "fc"]):
                param.requires_grad = True

    return model
