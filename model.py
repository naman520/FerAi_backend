from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import logging
import threading
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, support_credentials=True)
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more detailed logs

# Global variable for the model
model = None

# Load model in a separate thread
def load_model_thread():
    global model
    try:
        model = load_model('model_fer2013_v2.h5', compile=False)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")

# Start model loading in background
threading.Thread(target=load_model_thread).start()

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'status': 'ok',
        'message': 'Emotion Detection API is running'
    })

@app.route('/health', methods=['GET'])
def health_check():
    logging.debug("Health check endpoint called")  # Add debug log
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    logging.debug("Predict endpoint called")  # Add debug log
    if model is None:
        return jsonify({'error': 'Model not yet loaded'}), 503
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    try:
        img_data = request.files['image'].read()
        img = Image.open(io.BytesIO(img_data))
        
        # Preprocess the image
        img = img.convert('L').resize((48, 48))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 48, 48, 1)
        
        prediction = model.predict(img_array)
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        result = {emotion: float(score) for emotion, score in zip(emotion_labels, prediction[0])}
        
        return jsonify(result)
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)