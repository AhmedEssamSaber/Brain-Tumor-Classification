import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

class CNN_TUMOR(nn.Module):
    def __init__(self, params):
        super(CNN_TUMOR, self).__init__()

        filters = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        dropout_rate = params["dropout_rate"]

        self.features = nn.Sequential(
            nn.Conv2d(params["shape_in"][0], filters, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(filters, filters * 2, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(filters * 2, filters * 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(filters * 4, filters * 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Computing the Flattened Size Automatically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *params["shape_in"])  # batch size 1
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, num_fc1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(num_fc1, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class ResNet152_TUMOR(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_backbone=True):
        super(ResNet152_TUMOR, self).__init__()
        if pretrained:
            self.model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet152(weights=None)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False    

        # Change the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)