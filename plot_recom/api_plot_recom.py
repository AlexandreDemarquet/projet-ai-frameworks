from flask import Flask, request,jsonify
from annoy import AnnoyIndex
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizerFast,DistilBertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tfidf = TfidfVectorizer(stop_words='english',max_features=576)
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
distilbert_model.eval()

df_films = pd.read_csv('data/movies_metadata.csv')
titles_list = df_films['original_title'].tolist()

all_plots = df_films[df_films['overview'].notna()]['overview'].tolist()
tfidf_matrix = tfidf.fit_transform(all_plots)  # fit globalement

title_overview_dict = df_films.set_index('original_title')['overview'].to_dict()


# Charger l'index
dimension_bow = 576  
index_bow = AnnoyIndex(dimension_bow, 'angular')
index_bow.load("annoy_index_bow.ann")  # Charger l'index pré-construit

dimension_distil = 768
index_distil= AnnoyIndex(dimension_distil, 'angular')
index_distil.load("annoy_index_distil.ann")  # Charger l'index pré-construit

app = Flask(__name__)



def search(index,query_vector, k=5):
    indices = index.get_nns_by_vector(query_vector, k)
    titles = [titles_list[idx] for idx in indices]
    return titles


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        plot = data.get("plot")
        model_type = data.get("model")

        if model_type == "bow":
            index=index_bow
            embeddings = tfidf.transform([plot]).toarray()
        else:
            index=index_distil
            tokens = distilbert_tokenizer(plot, truncation=True, padding="longest", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = distilbert_model(**tokens)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        query_vector = embeddings.flatten()

        # Récupère les chemins vers les images similaires
        titles = search(index,query_vector) 
        resumes = [title_overview_dict[t] for t in titles]


        return jsonify({
            "titles": titles,
            "plots" : resumes,
            "format": "str",
            "note": "Chaque titre est une string"
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(port=8080, debug=True, use_reloader=False, host="0.0.0.0")


