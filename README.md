# 📘 projet-ai-frameworks DION/GRIS/GAY/DEMARQUET

## 📐 Architecture du projet
L’architecture du projet repose sur trois conteneurs distincts, tous connectés au **même réseau Docker**.
- **`api/`** : API principale, hébergée dans un conteneur optimisé pour les performances **CUDA / NVIDIA GPU**, gérant :
  - la classification d’affiches par genre,
  - l'explicabilité de la classification d’affiches par genre avec 3 méthodes -> affichage de la smoothgrad map / shap map / lime map,
  - la recommandation de films à partir d’un **résumé textuel**. Cette recommandation utilise deux bases de données Annoy, stockées sur un drive et téléchargées lors de la création de l'image docker.
  Cette API est conçue pour des **temps d'inférence rapides** grâce à l’accélération matérielle.

- **`ihm/`** : Interface utilisateur développée avec **Streamlit**, permettant aux utilisateurs d’interagir simplement avec les deux APIs (via boutons, uploads, champs texte, etc.).

- **`db/`** : Une API indépendante dédiée à la **recommandation d'affiches similaires à partir d'une image**. Ce service utilise une **base de données Annoy** (Approximate Nearest Neighbors) contenant des embeddings d’affiches pré-indexés et une base de donnée d'affiche en .jpg. Elle est séparée pour :
  - isoler le traitement spécifique à la recherche d’image,
  - éviter de surcharger l’API principale avec la base de donnée.
Il est tout à fait possible de regrouper l’ensemble des fonctionnalités (classification, recommandations textuelles et visuelles) dans **une seule API monolithique**. Cela simplifierait le déploiement et la gestion du projet.

Une architecture unifiée des deux apis est aussi envisagable mais dans notre cas, nous avons choisi de **séparer l’API dédiée à la recommandation d’affiches par image** pour des raisons de clarté et de séparation des fonctionalités. 

Néanmoins, toutes les APIs sont intégrées dans l’IHM de manière transparente, et l’utilisateur final ne perçoit aucune séparation fonctionnelle.

## 🔧 Fonctionnalités
-  **Classification de films par genre à partir d'une affiche** : envoie une image d'affiche, l'API prédit son genre (action, comédie, drame, etc.) + **explicabilité** via **SmoothGrad**, **SHAP** et **LIME** pour interpréter les décisions du modèle.
-  **Recommandation d'affiches de films similaires** : envoie une affiche, l'API renvoie des affiches visuellement proches.
-  **Recommandation de films à partir d’un résumé** : donne un synopsis textuel, l’API suggère des films similaires.

⚠️ **Remarque** : la fonctionnalité de recommandation d'affiches similaires repose sur une **API spécifique** connectée à une **base de données d'affiches existantes**.


## ⚙️ Installation & Lancement

### 🔁 Prérequis

- Carte graphique NVIDIA

### 🚀 Étapes d’installation

1. **Cloner le projet**

```bash
git clone https://github.com/AlexandreDemarquet/projet-ai-frameworks.git
cd projet-ai-frameworks
````

2. **Lancer les services avec Docker Compose**

```bash
docker-compose up --build
```

> ⚠️ La première exécution peut prendre du temps si les images doivent être construites.


### 🔍 Accès aux services

* **Interface utilisateur (IHM)** : [http://localhost:7860](http://localhost:7860)

## ✅ Tests & compatibilité

Le projet a été entièrement développé et testé dans un environnement **Linux Ubuntu 22.04** / **Ubuntu MATE 22.04**.

Chaque membre du projet a utilisé l'ordinateur fourni par l'école, équipé d'une **carte graphique NVIDIA RTX A500**.

L’API principale a donc été optimisée pour tourner efficacement avec le support GPU via Docker (avec `--gpus all`).




