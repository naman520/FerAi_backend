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
CORS(app)
logging.basicConfig(level=logging.INFO)

# Load model in a separate thread to avoid blocking
import tensorflow as tf
def load_model_thread():
       global model
       try:
           # Option 1: Try loading with object_compile set to False
           model = tf.keras.models.load_model('model_fer2013.h5', compile=False)
           print("Model loaded successfully")
       except Exception as e:
           print(f"Failed to load model with compile=False: {str(e)}")
           try:
               # Option 2: Try loading with custom_objects
               model = tf.keras.models.load_model('model_fer2013.h5', 
                                                  custom_objects={'InputLayer': tf.keras.layers.InputLayer})
               print("Model loaded successfully with custom objects")
           except Exception as e:
               print(f"Failed to load model with custom objects: {str(e)}")
               try:
                   # Option 3: Recreate the model structure
                   base_model = tf.keras.models.load_model('model_fer2013.h5', compile=False)
                   inputs = tf.keras.Input(shape=(48, 48, 1))
                   x = base_model(inputs)
                   model = tf.keras.Model(inputs, x)
                   print("Model reconstructed successfully")
               except Exception as e:
                   print(f"Failed to reconstruct model: {str(e)}")
                   raise

threading.Thread(target=load_model_thread).start()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img_data = request.files['image'].read()
        img = Image.open(io.BytesIO(img_data))
        
        # Preprocess the image (adjust based on your model's requirements)
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
