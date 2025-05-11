from annoy import AnnoyIndex
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizerFast,DistilBertModel

class MoviesDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['overview'].notna()]  # retire les NaN
        self.plots = self.data['overview'].tolist()

    def __len__(self):
        return len(self.plots)

    def __getitem__(self, idx):
        return self.plots[idx]
    
dataset = MoviesDataset("data/movies_metadata.csv")
all_plots = list(dataset.plots)

dataloader = DataLoader(dataset, batch_size=128, num_workers=0, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfidf = TfidfVectorizer(stop_words='english', max_features=576)
tfidf_matrix = tfidf.fit_transform(all_plots)  # fit globalement
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
distilbert_model.eval()


# Création des embeddings
features_list_bow = []
features_list_distil = []

plot_list=[]


# Compteur pour suivre l'image
print("Création des index de la db annoy")
for x in tqdm(dataloader):
    batch_size = len(x)
    plot_list.extend(x)

    # Embeddings BOW : extraire les lignes correspondantes du tfidf_matrix
    batch_indices = range(len(features_list_bow), len(features_list_bow) + batch_size)
    for i in batch_indices:
        vec = tfidf_matrix[i].toarray()[0]
        features_list_bow.append(vec)

    # Embeddings DistilBERT
    tokens = distilbert_tokenizer(x, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = distilbert_model(**tokens)
        cls_embeddings = outputs.last_hidden_state.mean(dim=1)
        features_list_distil.extend(cls_embeddings.cpu().numpy())



# Sauvegarde dans un CSV
df = pd.DataFrame({
    'plot': plot_list,
    'features_bow': [vec.tolist() for vec in features_list_bow],
    'features_distil': [vec.tolist() for vec in features_list_distil]
})
df.to_csv('annoy-database.csv', index=False)

# Sauvegarde de l'index Annoy BOW
dim_bow = len(features_list_bow[0])
annoy_index_bow = AnnoyIndex(dim_bow, 'angular')
for i, vec in enumerate(features_list_bow):
    annoy_index_bow.add_item(i, vec)
annoy_index_bow.build(10)
annoy_index_bow.save("annoy_index_bow.ann")

# Sauvegarde de l'index Annoy DistilBERT
dim_distil = len(features_list_distil[0])
annoy_index_distil = AnnoyIndex(dim_distil, 'angular')
for i, vec in enumerate(features_list_distil):
    annoy_index_distil.add_item(i, vec)
annoy_index_distil.build(10)
annoy_index_distil.save("annoy_index_distil.ann")
