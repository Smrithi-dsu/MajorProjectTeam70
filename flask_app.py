import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})

# Define the number of classes
num_classes = 6

# Load the MobileNetV2 model
mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
mobilenet_v2.classifier[1] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(mobilenet_v2.classifier[1].in_features, num_classes)
)

# Load the trained weights
model_path = 'best_model_kaggle.pth'
try:
    mobilenet_v2.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only = True))
    mobilenet_v2.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.content_type.startswith('image/'):
        return jsonify({"error": "Uploaded file is not an image"}), 400

    # Save and preprocess the image
    image_path = "./temp_image.jpg"
    try:
        file.save(image_path)
        img_tensor = preprocess_image(image_path)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

    # Perform prediction
    try:
        with torch.no_grad():
            predictions = mobilenet_v2(img_tensor)
        softmax_scores = torch.nn.functional.softmax(predictions[0], dim=0)
        predicted_class = torch.argmax(softmax_scores).item()
        confidence = softmax_scores[predicted_class].item()
        predicted_label = class_names[predicted_class]
        return jsonify({"prediction": predicted_label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)