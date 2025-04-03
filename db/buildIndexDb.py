from annoy import AnnoyIndex
import numpy as np
import os

# Paramètres de l'index
dimension = 100  # Nombre de dimensions des vecteurs
n_trees = 10     # Nombre d'arbres pour la recherche

# Création de l'index
index = AnnoyIndex(dimension, 'angular')

# Ajout de données fictives
np.random.seed(42)
for i in range(1000):
    vector = np.random.rand(dimension).astype(np.float32)
    index.add_item(i, vector)

# Construction de l'index
index.build(n_trees)

# Sauvegarde de l'index
index_path = "annoy_index.ann"
index.save(index_path)

print(f"Index Annoy sauvegardé dans {index_path}")
