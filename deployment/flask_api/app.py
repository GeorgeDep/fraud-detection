"""
Flask API for Fraud Detection with Grad-CAM Explanations
Simple REST API that demonstrates fraud detection capabilities
"""

import os
import io
import base64
import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template_string
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Initialize Flask app
app = Flask(__name__)

# Global variables for model
model = None
gradcam = None

# Model configuration
MODEL_CONFIG = {
    'model_name': 'efficientnet_b0.ra_in1k',
    'num_classes': 2,
    'input_size': (224, 224),
    'class_names': ['Authentic', 'Tampered']
}

# Image preprocessing transforms
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(MODEL_CONFIG['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Denormalization for visualization
DENORMALIZE = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


def load_fraud_detection_model(model_path: str):
    """Load the trained fraud detection model"""
    global model, gradcam
    
    print(f"Loading fraud detection model from {model_path}")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load complete model
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        model.to(device)
        print("Model loaded successfully")
        
        # Setup Grad-CAM
        target_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layers = [module]
        
        if target_layers:
            gradcam = GradCAM(model=model, target_layers=target_layers)
            print("Grad-CAM initialized successfully")
        else:
            print("Warning: No convolutional layers found for Grad-CAM")
            gradcam = None
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess PIL image for model inference"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    tensor = INFERENCE_TRANSFORMS(image)
    return tensor


def predict_fraud(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Generate fraud prediction for input image"""
    global model
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Get model output
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': MODEL_CONFIG['class_names'][predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                'authentic': float(probabilities[0][0]),
                'tampered': float(probabilities[0][1])
            },
            'risk_score': float(probabilities[0][1]),
            'is_fraudulent': predicted_class == 1
        }
        
        # Add interpretation
        if predicted_class == 1:  # Tampered
            if confidence > 0.9:
                interpretation = f"HIGH FRAUD RISK: Model detected tampering with {confidence:.1%} confidence. Immediate investigation recommended."
            elif confidence > 0.7:
                interpretation = f"MODERATE FRAUD RISK: Model detected likely tampering with {confidence:.1%} confidence. Manual review advised."
            else:
                interpretation = f"LOW FRAUD RISK: Model detected possible tampering with {confidence:.1%} confidence. Consider additional verification."
        else:  # Authentic
            if confidence > 0.9:
                interpretation = f"AUTHENTIC: Document appears genuine with {confidence:.1%} confidence. Low fraud risk."
            else:
                interpretation = f"LIKELY AUTHENTIC: Document appears genuine with {confidence:.1%} confidence. Some uncertainty detected."
        
        result['interpretation'] = interpretation
        
        return result


def generate_gradcam_explanation(image_tensor: torch.Tensor, predicted_class: int) -> Dict[str, Any]:
    """Generate Grad-CAM explanation for the prediction"""
    global gradcam
    
    if gradcam is None:
        return {
            'explanation_available': False,
            'error': 'Grad-CAM not available'
        }
    
    try:
        # Generate Grad-CAM
        targets = [ClassifierOutputTarget(predicted_class)]
        input_tensor = image_tensor.unsqueeze(0)
        
        grayscale_cam = gradcam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Create visualization
        original_image = tensor_to_numpy(image_tensor)
        cam_overlay = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
        
        # Calculate explanation metrics
        high_activation_ratio = np.sum(grayscale_cam > 0.7) / grayscale_cam.size
        avg_intensity = np.mean(grayscale_cam)
        max_activation = np.max(grayscale_cam)
        
        # Convert visualizations to base64 for JSON response
        heatmap_b64 = array_to_base64(grayscale_cam, colormap='jet')
        overlay_b64 = array_to_base64(cam_overlay)
        original_b64 = array_to_base64((original_image * 255).astype(np.uint8))
        
        explanation_result = {
            'explanation_available': True,
            'target_class': predicted_class,
            'target_label': MODEL_CONFIG['class_names'][predicted_class],
            'heatmap_intensity': float(avg_intensity),
            'max_activation': float(max_activation),
            'high_activation_ratio': float(high_activation_ratio),
            'visualizations': {
                'original_image': original_b64,
                'heatmap': heatmap_b64,
                'overlay': overlay_b64
            }
        }
        
        return explanation_result
        
    except Exception as e:
        return {
            'explanation_available': False,
            'error': str(e)
        }


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor to numpy array for visualization"""
    denorm_tensor = DENORMALIZE(tensor)
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
    image_np = denorm_tensor.permute(1, 2, 0).numpy()
    return image_np.astype(np.float32)


def array_to_base64(array: np.ndarray, colormap: str = None) -> str:
    """Convert numpy array to base64 encoded image"""
    try:
        if colormap and len(array.shape) == 2:
            # Apply colormap to grayscale heatmap
            import matplotlib.cm as cm
            colored_array = cm.get_cmap(colormap)(array)
            colored_array = (colored_array[:, :, :3] * 255).astype(np.uint8)
            image = Image.fromarray(colored_array)
        else:
            # Regular image array
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            image = Image.fromarray(array)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error converting array to base64: {e}")
        return ""


# API Routes
@app.route('/')
def home():
    """API documentation page"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }
            .method { color: #007acc; font-weight: bold; }
            code { background: #eee; padding: 2px 6px; }
        </style>
    </head>
    <body>
        <h1>🔍 Fraud Detection API</h1>
        <p>AI-powered document fraud detection with explainable AI capabilities.</p>
        
        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /health</h3>
            <p>Check API health status</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /predict</h3>
            <p>Detect fraud in document images</p>
            <p><strong>Body:</strong> <code>{"image_base64": "..."}</code></p>
            <p><strong>Returns:</strong> Prediction with confidence scores and interpretation</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /explain</h3>
            <p>Get fraud prediction with Grad-CAM visual explanations</p>
            <p><strong>Body:</strong> <code>{"image_base64": "..."}</code></p>
            <p><strong>Returns:</strong> Prediction + Grad-CAM heatmaps and overlays</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /demo</h3>
            <p>Run demonstration with test images</p>
        </div>
        
        <h2>Model Information</h2>
        <ul>
            <li><strong>Architecture:</strong> EfficientNet-B0</li>
            <li><strong>Task:</strong> Binary classification (Authentic vs Tampered)</li>
            <li><strong>Interpretability:</strong> Grad-CAM explanations</li>
            <li><strong>Input:</strong> RGB images (224x224)</li>
        </ul>
    </body>
    </html>
    """
    return html_template


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'gradcam_available': gradcam is not None,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    return jsonify(status)


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Fraud detection prediction endpoint"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'No image_base64 provided'}), 400
        
        # Decode image
        image_data = base64.b64decode(data['image_base64'])
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess and predict
        image_tensor = preprocess_image(image)
        prediction = predict_fraud(image_tensor)
        
        # Add metadata
        prediction['api_version'] = '1.0'
        prediction['processing_time'] = time.time()
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain', methods=['POST'])
def explain_endpoint():
    """Fraud detection with Grad-CAM explanations"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'No image_base64 provided'}), 400
        
        # Decode image
        image_data = base64.b64decode(data['image_base64'])
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess and predict
        image_tensor = preprocess_image(image)
        prediction = predict_fraud(image_tensor)
        
        # Generate Grad-CAM explanation
        explanation = generate_gradcam_explanation(image_tensor, prediction['predicted_class'])
        
        # Combine results
        result = {
            'prediction': prediction,
            'explanation': explanation,
            'api_version': '1.0',
            'processing_time': time.time()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/demo', methods=['GET'])
def demo_endpoint():
    """Run demonstration with test images"""
    try:
        # Find test images in current directory
        test_images = []
        for pattern in ['test_authentic_*.jpg', 'test_tampered_*.jpg']:
            test_images.extend(list(Path('.').glob(pattern)))
        
        if not test_images:
            return jsonify({'error': 'No test images found in current directory'}), 404
        
        results = []
        
        for img_path in test_images[:4]:  # Limit to 4 images for demo
            try:
                # Load and encode image
                with open(img_path, 'rb') as f:
                    image_data = f.read()
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                
                # Get prediction with explanation
                image = Image.open(img_path)
                image_tensor = preprocess_image(image)
                prediction = predict_fraud(image_tensor)
                explanation = generate_gradcam_explanation(image_tensor, prediction['predicted_class'])
                
                results.append({
                    'image_name': img_path.name,
                    'prediction': prediction,
                    'explanation': explanation
                })
                
            except Exception as e:
                results.append({
                    'image_name': img_path.name,
                    'error': str(e)
                })
        
        return jsonify({
            'demo_results': results,
            'total_images': len(results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Configuration
    MODEL_PATH = 'fraud_detection_complete_20250603_145722.pth'
    
    print("=" * 60)
    print("FRAUD DETECTION API - FLASK DEPLOYMENT")
    print("=" * 60)
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please ensure the model file is in the current directory")
        exit(1)
    
    # Load model
    print("🔄 Loading fraud detection model...")
    if not load_fraud_detection_model(MODEL_PATH):
        print("❌ Failed to load model")
        exit(1)
    
    print("✅ Model loaded successfully")
    print("✅ Grad-CAM explanations ready")
    
    # Install flask if needed
    try:
        import flask
    except ImportError:
        print("Installing Flask...")
        os.system("pip3 install flask")
        import flask
    
    print("\n🚀 Starting Fraud Detection API...")
    print("📍 API will be available at: http://localhost:5000")
    print("📋 Documentation: http://localhost:5000")
    print("🔍 Health check: http://localhost:5000/health")
    print("\n" + "=" * 60)
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)