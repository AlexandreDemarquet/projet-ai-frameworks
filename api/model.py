import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class FilmClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FilmClassifier, self).__init__()

        # Charger resnet pré-entraîné
        self.model = models.resnet50(pretrained=True)

        # Adapter la dernière couche au nombre de classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def forward(self, x):
        return self.model(x)