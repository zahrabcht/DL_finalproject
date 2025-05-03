from flask import Flask, request, jsonify, render_template  # type: ignore
import tensorflow as tf  # type: ignore
import numpy as np
import os
import cv2  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

app = Flask(__name__)

# Chemin absolu vers le modèle
model_path = '/home/zahra-boucheta/Documents/projet_dl_f/ferNet.h5'

# Vérification d'existence du fichier modèle
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier modèle est introuvable à : {model_path}")

# Chargement du modèle
model = tf.keras.models.load_model(model_path)

# Classes d’émotions (ordre exact selon l'entraînement)
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def preprocess_image(image_path):
    """Prétraitement de l'image avant prédiction."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # (48, 48, 1)
    img = np.expand_dims(img, axis=0)   # (1, 48, 48, 1)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier n’a été envoyé'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'})

    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    img = preprocess_image(file_path)
    prediction = model.predict(img)

    # Meilleure prédiction
    top_idx = int(np.argmax(prediction))
    emotion = emotion_classes[top_idx]
    confidence = float(np.max(prediction))

    return jsonify({
        'emotion': emotion,
        'confidence': round(confidence, 2),
        'probabilities': {
            emotion_classes[i]: round(float(prediction[0][i]), 2)
            for i in range(len(emotion_classes))
        }
    })

if __name__ == '__main__':
    app.run(debug=True) 