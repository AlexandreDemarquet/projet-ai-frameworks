from annoy import AnnoyIndex
import numpy as np
import pandas as pd
import os
import torch
import torchvision
import torchvision.models as models
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# Chargement des données d'images
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]
normalize = transforms.Normalize(mean, std)
inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(mean, std)],
   std= [1/s for s in std]
)

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                normalize])
dataset = torchvision.datasets.ImageFolder(root="/sorted_movie_posters_paligema", transform=transform)

dataloader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)


# Chargement du modèle MobileNetV3 pré-entraîné pour générer des embeddings
mobilenet = models.mobilenet_v3_small(pretrained=True)

model = torch.nn.Sequential(mobilenet.features, mobilenet.avgpool, torch.nn.Flatten()).cuda()


# Création des embeddings par extraction des caractéristiques
features_list = []
paths_list = []
for x, paths in tqdm(dataloader):
    with torch.no_grad():
        embeddings = model(x.cuda())
        features_list.extend(embeddings.cpu().numpy())
        paths_list.extend(paths)


# Création et sauvegarde d'un index Annoy à partir de fichiers d'images
dim = 576
annoy_index = AnnoyIndex(dim, 'angular')
for i, embedding in enumerate(features_list):
    annoy_index.add_item(i, embedding)

annoy_index.build(10)

# Sauvegarde de l'index
index_path = "annoy_index.ann"
annoy_index.save(index_path)

print(f"Index Annoy sauvegardé dans {index_path}")
