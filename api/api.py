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

app = Flask(__name__)

# ===================== CONFIGURATION =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- TEXT MODEL SETUP --------
tfidf = TfidfVectorizer(stop_words='english', max_features=576)
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
distilbert_model.eval()

df_films = pd.read_csv('data/movies_metadata.csv')
titles_list = df_films['original_title'].tolist()
all_plots = df_films[df_films['overview'].notna()]['overview'].tolist()
tfidf_matrix = tfidf.fit_transform(all_plots)
title_overview_dict = df_films.set_index('original_title')['overview'].to_dict()

index_bow = AnnoyIndex(576, 'angular')
index_bow.load("annoy_index_bow.ann")

index_distil = AnnoyIndex(768, 'angular')
index_distil.load("annoy_index_distil.ann")

# -------- IMAGE MODEL SETUP --------
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='weights/filmClassifier.pth', help='path of the model')
args, unknown = parser.parse_known_args()  # pour éviter les erreurs en Flask

model = FilmClassifier(10)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ===================== ROUTES =====================

def search(index, query_vector, k=5):
    indices = index.get_nns_by_vector(query_vector, k)
    titles = [titles_list[idx] for idx in indices]
    return titles

@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        data = request.get_json()
        plot = data.get("plot")
        model_type = data.get("model")

        if model_type == "bow":
            index = index_bow
            embeddings = tfidf.transform([plot]).toarray()
        else:
            index = index_distil
            tokens = distilbert_tokenizer(plot, truncation=True, padding="longest", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = distilbert_model(**tokens)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        query_vector = embeddings.flatten()
        titles = search(index, query_vector)
        resumes = [title_overview_dict[t] for t in titles]

        return jsonify({
            "titles": titles,
            "plots": resumes,
            "format": "str",
            "note": "Chaque titre est une string"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_image', methods=['POST'])
def predict_image():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary)).convert("RGB")
    tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = outputs.max(1)

    return jsonify({"prediction": int(predicted[0])})

@app.route('/batch_predict_image', methods=['POST'])
def batch_predict_image():
    images_binary = request.files.getlist("images[]")
    tensors = []

    for img_binary in images_binary:
        img_pil = Image.open(img_binary.stream).convert("RGB")
        tensor = transform(img_pil)
        tensors.append(tensor)

    batch_tensor = torch.stack(tensors, dim=0).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)
        _, predictions = outputs.max(1)

    return jsonify({"predictions": predictions.tolist()})

def get_vanilla_grad(img, model):
    img.retain_grad()
    output = model(img)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()
    return img.grad

@app.route('/smoothgrad', methods=['POST'])
def smoothgrad():
    img_file = request.data
    img_pil = Image.open(io.BytesIO(img_file)).convert("RGB")
    np_img = np.array(img_pil)

    tensor = transform(img_pil).to(device).unsqueeze(0)
    tensor.requires_grad_()

    stdev_spread = 0.15
    n_samples = 100
    stdev = stdev_spread * (tensor.max() - tensor.min())
    total_gradients = torch.zeros_like(tensor, device=device)

    for _ in range(n_samples):
        noise = np.random.normal(0, stdev.item(), tensor.shape).astype(np.float32)
        noisy_img = tensor + torch.tensor(noise, device=device, requires_grad=True)
        grad = get_vanilla_grad(noisy_img, model)
        total_gradients += grad * grad

    total_gradients /= n_samples
    saliency, _ = torch.max(total_gradients.abs(), dim=1)
    saliency = saliency.squeeze(0).cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(np_img)
    ax[0].axis('off')
    ax[1].imshow(saliency, cmap='hot')
    ax[1].axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

lime_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.route('/lime', methods=['POST'])
def lime_explanation():
    try:
        img_file = request.data
        img_pil = Image.open(io.BytesIO(img_file)).convert("RGB")
        np_img = np.array(img_pil)

        def batch_predict(images):
            model.eval()
            batch = torch.stack([lime_transform(Image.fromarray(img)) for img in images], dim=0).to(device)
            with torch.no_grad():
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np_img, batch_predict, top_labels=5, hide_color=0, num_samples=1000
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False
        )
        img_boundary = mark_boundaries(temp / 255.0, mask)

        fig, ax = plt.subplots()
        ax.imshow(img_boundary)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===================== MAIN =====================
if __name__ == "__main__":
    app.run(port=5000, debug=True, host="0.0.0.0")
