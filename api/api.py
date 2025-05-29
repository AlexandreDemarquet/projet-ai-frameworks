import argparse
import io
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_file
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizerFast, DistilBertModel
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from model import FilmClassifier
import shap

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== ARGUMENTS =====================
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='weights/filmClassifier.pth', help='path of the model')
args, unknown = parser.parse_known_args()

# ===================== MODELS =====================
model = FilmClassifier(10)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

tfidf = TfidfVectorizer(stop_words='english', max_features=576)
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
distilbert_model.eval()

# ===================== DATA =====================
df_films = pd.read_csv('data/movies_metadata.csv')
titles_list = df_films['original_title'].tolist()
all_plots = df_films[df_films['overview'].notna()]['overview'].tolist()
tfidf_matrix = tfidf.fit_transform(all_plots)
title_overview_dict = df_films.set_index('original_title')['overview'].to_dict()

index_bow = AnnoyIndex(576, 'angular')
index_bow.load("annoy_index_bow.ann")

index_distil = AnnoyIndex(768, 'angular')
index_distil.load("annoy_index_distil.ann")

# ===================== TRANSFORMS =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

lime_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ===================== FUNCTIONS =====================

# Prédit les films similaires à un texte donné (en utilisant BOW ou DistilBERT)
@app.route('/predict_text', methods=['POST'])
def predict_text():
    data = request.get_json()
    plot = data['plot']
    model_type = data['model']

    if model_type == 'bow':
        plot_vector = tfidf.transform([plot]).toarray()[0]
        index = index_bow
    else:
        inputs = distilbert_tokenizer(plot, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
        plot_vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        index = index_distil

    similar_indices = index.get_nns_by_vector(plot_vector, 5)
    recommended_titles = [titles_list[i] for i in similar_indices]
    return jsonify(recommended_titles)

# Prédit la classe d'un poster image
@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1).cpu().numpy().tolist()[0]
    return jsonify(probabilities)

# Prédit des classes pour plusieurs images à la fois
@app.route('/batch_predict_image', methods=['POST'])
def batch_predict_image():
    images = request.files.getlist('images')
    tensors = [transform(Image.open(file.stream).convert('RGB')).unsqueeze(0) for file in images]
    batch = torch.cat(tensors).to(device)

    with torch.no_grad():
        output = model(batch)
        probabilities = F.softmax(output, dim=1).cpu().numpy().tolist()
    return jsonify(probabilities)

# Produit une explication SmoothGrad pour une image
@app.route('/smoothgrad', methods=['POST'])
def smoothgrad():
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device).requires_grad_()

    noise_level = 0.2
    n_samples = 50
    grads = []
    for _ in range(n_samples):
        noisy_img = image_tensor + noise_level * torch.randn_like(image_tensor)
        noisy_img.requires_grad_()
        output = model(noisy_img)
        output[0, output.argmax()].backward(retain_graph=True)
        grads.append(noisy_img.grad.detach())

    avg_grad = torch.mean(torch.stack(grads), dim=0).squeeze().cpu().numpy()
    avg_grad = np.transpose(avg_grad, (1, 2, 0))
    avg_grad = (avg_grad - avg_grad.min()) / (avg_grad.max() - avg_grad.min())
    avg_grad = (avg_grad * 255).astype(np.uint8)
    image_pil = Image.fromarray(avg_grad)
    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')

# Produit une explication LIME pour une image
@app.route('/lime', methods=['POST'])
def lime_explanation():
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')

    def batch_predict(images):
        batch = torch.stack([lime_transform(Image.fromarray(img)).to(device) for img in images])
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(image), batch_predict, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    lime_img = mark_boundaries(temp / 255.0, mask)
    lime_img = (lime_img * 255).astype(np.uint8)
    image_pil = Image.fromarray(lime_img)
    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')

# Produit une explication SHAP pour une image
@app.route('/shap', methods=['POST'])
def shap_explanation():
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    background = torch.cat([img_tensor for _ in range(5)], dim=0)
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(img_tensor)

    shap_img = np.abs(shap_values[0][0]).mean(0)
    shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min())
    shap_img = (shap_img * 255).astype(np.uint8)
    image_pil = Image.fromarray(shap_img)
    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')

# Renvoie les indices des films similaires avec BOW ou DistilBERT
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    plot = data['plot']
    model_type = data['model']

    if model_type == 'bow':
        plot_vector = tfidf.transform([plot]).toarray()[0]
        index = index_bow
    else:
        inputs = distilbert_tokenizer(plot, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
        plot_vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        index = index_distil

    similar_indices = index.get_nns_by_vector(plot_vector, 5)
    return jsonify(similar_indices)

# Pareil que predict mais renvoie plusieurs réponses
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.get_json()
    plots = data['plots']
    model_type = data['model']

    results = []
    for plot in plots:
        if model_type == 'bow':
            plot_vector = tfidf.transform([plot]).toarray()[0]
            index = index_bow
        else:
            inputs = distilbert_tokenizer(plot, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = distilbert_model(**inputs)
            plot_vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            index = index_distil

        similar_indices = index.get_nns_by_vector(plot_vector, 5)
        results.append(similar_indices)
    return jsonify(results)

# ===================== MAIN =====================
if __name__ == "__main__":
    app.run(port=5000, debug=True, host="0.0.0.0")
