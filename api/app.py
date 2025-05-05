import argparse
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import io
from model import FilmClassifier
import matplotlib.pyplot as plt
import numpy as np
from flask import send_file


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/weights/filmClassifier.pth', help='path of the model') # add an argument '--model_path'
args = parser.parse_args()
model_path = args.model_path

model = FilmClassifier(10)
# Load the model
model.load_state_dict(torch.load(model_path, weights_only=False))
model.to(device)
model.eval()

# Les différentes catégories : {'action': 0, 'animation': 1, 'comedy': 2, 'documentary': 3, 
# 'drama': 4, 'fantasy': 5, 'horror': 6,
#  'romance': 7, 'science Fiction': 8, 'thriller': 9}


transform = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

@app.route('/predict', methods=['POST'])
def predict():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))

    # Transform the PIL image
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = outputs.max(1)

    return jsonify({"prediction": int(predicted[0])})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    # Get the image data from the request
    images_binary = request.files.getlist("images[]")

    tensors = []

    for img_binary in images_binary:
        img_pil = Image.open(img_binary.stream)
        tensor = transform(img_pil)
        tensors.append(tensor)

    # Stack tensors to form a batch tensor
    batch_tensor = torch.stack(tensors, dim=0)

    # Make prediction
    with torch.no_grad():
        outputs = model(batch_tensor.to(device))
        _, predictions = outputs.max(1)

    return jsonify({"predictions": predictions.tolist()})

def get_vanilla_grad(img, model):
    img.retain_grad()
    output = model(img)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()
    return img.grad

@app.route('/saliency', methods=['POST'])
def saliency():
    # Récupérer l'image
    img_file = request.data #request.files["image"]
    # img_pil = Image.open(img_file.stream).convert("RGB")
    # np_img = np.array(img_pil)

    # Transformer en tenseur
    # img = transform(img_pil).unsqueeze(0).to(device)
    img_pil = Image.open(io.BytesIO(img_file))
    np_img = np.array(img_pil)

    # Transform the PIL image
    tensor = transform(img_pil).to(device)
    img = tensor.unsqueeze(0)
    img.requires_grad_()

    # Calcul des gradients avec bruit
    stdev_spread = 0.15
    n_samples = 100
    stdev = stdev_spread * (img.max() - img.min())
    total_gradients = torch.zeros_like(img, device='cuda')

    for _ in range(n_samples):
        noise = np.random.normal(0, stdev.item(), img.shape).astype(np.float32)
        noisy_img = img + torch.tensor(noise, device='cuda', requires_grad=True)
        grad = get_vanilla_grad(noisy_img, model)
        total_gradients += grad * grad

    total_gradients /= n_samples
    saliency, _ = torch.max(total_gradients.abs(), dim=1)
    saliency = saliency.squeeze(0).cpu().numpy()

    # Créer l’image de la saliency map
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(np_img)
    ax[0].axis('off')
    ax[1].imshow(saliency, cmap='hot')
    ax[1].axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return send_file(buf, mimetype='image/png')


if __name__ == "__main__":
    app.run(port=5000, debug=True, host="0.0.0.0")
