from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import logging
import threading
import os
import time

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}}, support_credentials=True)
logging.basicConfig(level=logging.DEBUG)

# Global variables
model = None
model_loading = False
model_load_time = None
last_prediction_time = None
prediction_count = 0

def load_model_thread():
    global model, model_loading, model_load_time
    model_loading = True
    try:
        model = load_model('model_fer2013_v2.h5', compile=False)
        model_load_time = time.time()
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
    finally:
        model_loading = False

# Start model loading in background
threading.Thread(target=load_model_thread).start()

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'status': 'ok',
        'message': 'Emotion Detection API is running'
    })

@app.route('/predict', methods=['POST'])
def predict():
    global last_prediction_time, prediction_count
    
    # Include health check information in the response
    health_status = {
        'model_loaded': model is not None,
        'model_loading': model_loading,
        'uptime': time.time() - model_load_time if model_load_time else None,
        'predictions_made': prediction_count,
        'last_prediction': time.time() - last_prediction_time if last_prediction_time else None
    }
    
    # If model is still loading, return status with 503
    if model is None:
        if model_loading:
            return jsonify({
                'error': 'Model is still loading',
                'health_status': health_status
            }), 503
        else:
            return jsonify({
                'error': 'Model failed to load',
                'health_status': health_status
            }), 500
    
    # Check if image file is provided
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file provided',
            'health_status': health_status
        }), 400
        
    try:
        start_time = time.time()
        
        # Process the image
        img_data = request.files['image'].read()
        img = Image.open(io.BytesIO(img_data))
        
        # Preprocess the image
        img = img.convert('L').resize((48, 48))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 48, 48, 1)
        
        # Make prediction
        prediction = model.predict(img_array)
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        result = {emotion: float(score) for emotion, score in zip(emotion_labels, prediction[0])}
        
        # Update metrics
        last_prediction_time = time.time()
        prediction_count += 1
        processing_time = time.time() - start_time
        
        # Return combined result with health information
        return jsonify({
            'predictions': result,
            'health_status': health_status,
            'processing_time': processing_time
        })
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'health_status': health_status
        }), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        feedback_data = request.json
        # Here you could add logic to store or process the feedback
        logging.info(f"Received feedback: {feedback_data}")
        return jsonify({'status': 'success', 'message': 'Feedback received'})
    except Exception as e:
        logging.error(f"Feedback error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)