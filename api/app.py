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
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torch.nn.functional as F
import shap



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/weights/filmClassifier.pth', help='path of the model') # add an argument '--model_path'
args = parser.parse_args()
model_path = args.model_path

model = FilmClassifier(10)

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

    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0) 

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = outputs.max(1)

    return jsonify({"prediction": int(predicted[0])})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    images_binary = request.files.getlist("images[]")

    tensors = []

    for img_binary in images_binary:
        img_pil = Image.open(img_binary.stream)
        tensor = transform(img_pil)
        tensors.append(tensor)

    batch_tensor = torch.stack(tensors, dim=0)

    with torch.no_grad():
        outputs = model(batch_tensor.to(device))
        _, predictions = outputs.max(1)

    return jsonify({"predictions": predictions.tolist()})


########### XAI #####################

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

    img_pil = Image.open(io.BytesIO(img_file))
    np_img = np.array(img_pil)

    tensor = transform(img_pil).to(device)
    img = tensor.unsqueeze(0)
    img.requires_grad_()

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


means = [0.5, 0.5, 0.5] 
stds = [0.5, 0.5, 0.5]

lime_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
])

@app.route('/lime', methods=['POST'])
def lime_explanation():
    try:
        img_file = request.data
        img_pil = Image.open(io.BytesIO(img_file))
        np_img = np.array(img_pil)

        def batch_predict(images):
            model.eval()
            batch = torch.stack([lime_transform(Image.fromarray(img)) for img in images], dim=0)
            batch = batch.to(device)
            with torch.no_grad():
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np_img,
            batch_predict,
            top_labels=5,
            hide_color=0,
            num_samples=1000
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=10,
            hide_rest=False
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

@app.route('/shap', methods=['POST'])
def shap_explanation():
    try:
        img_file = request.data
        img_pil = Image.open(io.BytesIO(img_file)).convert("RGB")
        np_img = np.array(img_pil)

        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)
            predicted_class = logits.argmax(dim=1).item()

        input_np = np.array(img_pil) / 255.0 

        masker = shap.maskers.Image("inpaint_telea", input_np.shape)

        def wrapped_model(X):
            X_torch = torch.tensor(X).permute(0, 3, 1, 2).float().to(device)
            with torch.no_grad():
                out = model(X_torch)
                probs = torch.nn.functional.softmax(out, dim=1)
            return probs.cpu().numpy()

        explainer = shap.Explainer(
            wrapped_model,
            masker=masker,
            output_names=[str(i) for i in range(logits.shape[1])]
        )

        shap_values = explainer(
            input_np[np.newaxis, :, :, :],
            max_evals=500,
            batch_size=20,
            outputs=shap.Explanation.argsort.flip[:1]
        )

        buf = io.BytesIO()
        shap.image_plot(shap_values, show=False)

        plt.gca().set_title('')
        plt.gca().set_xlabel('')
        plt.gca().set_ylabel('')

        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)

        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(port=5000, debug=True, host="0.0.0.0")
