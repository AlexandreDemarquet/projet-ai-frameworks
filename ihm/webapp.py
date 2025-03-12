import gradio as gr
from PIL import Image
import requests
import io

#model =

def recognize_genre(image):
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    # Send request to the API
    response = requests.post("http://127.0.0.1:5000/predict", data=img_binary.getvalue())
    predicted_label = response.json().get("prediction", "Genre inconnu")
    return predicted_label

def find_similar(image):
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")

    #charger le modèle d'embedding
    
    embedding = 0#model(img_binary.getvalue())

    # Send request to the API
    response = requests.post("http://127.0.0.1:5001/predict", data=embedding)

    #response est une liste de path, on récupère ensuite une liste d'image
    images = [Image.open(io.BytesIO(x)) for x in response.content]

    return images

def all(image):
    return recognize_genre(image),find_similar(image)

if __name__=='__main__':

    gr.Interface(fn=all, 
                inputs="image", 
                outputs=["text", "image"],
                live=True,
                description="Mettre un poster de film pour prédire son genre, et avoir des recommandations",
                ).launch(debug=True, share=True)