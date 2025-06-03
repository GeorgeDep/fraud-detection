"""
SageMaker inference script for fraud detection model
Handles model loading and prediction requests for local deployment
"""

import json
import logging
import os
import io
import base64
from typing import Dict, Any

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import timm

# Configure logging
logger = logging.getLogger(__name__)

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


def model_fn(model_dir):
    """
    Load the trained model for inference
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Loaded PyTorch model
    """
    logger.info(f"Loading model from {model_dir}")
    
    # Find model file
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError(f"No .pth model files found in {model_dir}")
    
    model_path = os.path.join(model_dir, model_files[0])
    logger.info(f"Loading model from {model_path}")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load complete model
        model = torch.load(model_path, map_location=device, weights_only=False)
        logger.info("Loaded complete model successfully")
    except Exception as e:
        logger.warning(f"Failed to load complete model: {e}")
        logger.info("Attempting to load state dict...")
        
        # Fallback: Load state dict and reconstruct model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = timm.create_model(
            MODEL_CONFIG['model_name'],
            pretrained=False,
            num_classes=MODEL_CONFIG['num_classes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Reconstructed model from state dict")
    
    model.eval()
    model.to(device)
    
    logger.info(f"Model loaded successfully on {device}")
    return model


def input_fn(request_body, content_type='application/json'):
    """
    Process input data for inference
    
    Args:
        request_body: Raw request body
        content_type: Content type of request
        
    Returns:
        Processed input data
    """
    logger.info(f"Processing input with content type: {content_type}")
    
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Handle base64 encoded image
        if 'image_base64' in input_data:
            image_data = base64.b64decode(input_data['image_base64'])
            image = Image.open(io.BytesIO(image_data))
        else:
            raise ValueError("No image data provided in request")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing transforms
        image_tensor = INFERENCE_TRANSFORMS(image)
        
        return {
            'image_tensor': image_tensor,
            'original_size': image.size
        }
    
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Generate prediction from processed input
    
    Args:
        input_data: Processed input from input_fn
        model: Loaded model from model_fn
        
    Returns:
        Prediction results
    """
    logger.info("Running prediction")
    
    device = next(model.parameters()).device
    image_tensor = input_data['image_tensor'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get model output
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Prepare response
        result = {
            'predicted_class': predicted_class,
            'predicted_label': MODEL_CONFIG['class_names'][predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                'authentic': float(probabilities[0][0]),
                'tampered': float(probabilities[0][1])
            },
            'risk_score': float(probabilities[0][1]),  # Tampered probability as risk
            'is_fraudulent': predicted_class == 1,
            'original_image_size': input_data['original_size']
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
        
        logger.info(f"Prediction complete: {result['predicted_label']} ({confidence:.3f})")
        return result


def output_fn(prediction, accept='application/json'):
    """
    Format prediction output
    
    Args:
        prediction: Prediction results from predict_fn
        accept: Accepted response format
        
    Returns:
        Formatted response
    """
    logger.info(f"Formatting output for accept type: {accept}")
    
    if accept == 'application/json':
        return json.dumps(prediction, indent=2)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# For local testing
if __name__ == "__main__":
    print("SageMaker inference script loaded successfully")
    print("This script provides fraud detection with the following capabilities:")
    print("- Binary classification: Authentic vs Tampered")
    print("- Confidence scoring and risk assessment")
    print("- Automated interpretation of results")
    print("- Support for various image formats")