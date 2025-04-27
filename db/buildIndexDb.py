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
from embeggings_model import model



# Chargement des données d'images
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean, std)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

dataset = torchvision.datasets.ImageFolder(root="sorted_movie_posters_paligema", transform=transform)
dataloader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)

# On récupère la liste complète des chemins
all_image_paths = [sample[0] for sample in dataset.samples]

# Chargement du modèle
model = model.cuda()

# Création des embeddings
features_list = []
paths_list = []

# Compteur pour suivre l'image
idx = 0

for x, _ in tqdm(dataloader):
    with torch.no_grad():
        embeddings = model(x.cuda())
        features_list.extend(embeddings.cpu().numpy())

    # Récupération des chemins pour ce batch
    batch_size = x.size(0)
    paths_batch = all_image_paths[idx: idx + batch_size]
    paths_list.extend(paths_batch)
    idx += batch_size

# Sauvegarde dans un CSV
df = pd.DataFrame({
    'features': features_list,
    'path': paths_list
})
df.to_csv('annoy-database.csv', index=False)

# Création et sauvegarde de l'index
dim = 576
annoy_index = AnnoyIndex(dim, 'angular')

for i, embedding in enumerate(features_list):
    annoy_index.add_item(i, embedding)

annoy_index.build(10)
annoy_index.save("annoy_index.ann")

print("Index Annoy sauvegardé.")
