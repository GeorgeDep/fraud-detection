"""
Test and demonstrate fraud detection endpoint with Grad-CAM explanations
Shows API consumption and generates visual explanations separately
"""

import json
import base64
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from deploy_local import deploy_fraud_detection_model


class FraudDetectionAPITester:
    """
    Test suite for fraud detection API with Grad-CAM explanations
    """
    
    def __init__(self, predictor, model_path: str):
        """
        Initialize tester with deployed predictor and model for Grad-CAM
        
        Args:
            predictor: Deployed SageMaker predictor
            model_path: Path to model file for Grad-CAM
        """
        self.predictor = predictor
        self.model_path = model_path
        self.gradcam_model = self._load_gradcam_model()
        self.gradcam = self._setup_gradcam()
        
        # Image preprocessing for Grad-CAM
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("API tester initialized with Grad-CAM capability")
    
    def _load_gradcam_model(self):
        """Load model for Grad-CAM explanations"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(self.model_path, map_location=device, weights_only=False)
        model.eval()
        return model
    
    def _setup_gradcam(self):
        """Setup Grad-CAM for explanations"""
        # Find the last convolutional layer
        target_layers = []
        for name, module in self.gradcam_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layers = [module]
        
        return GradCAM(model=self.gradcam_model, target_layers=target_layers)
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image file to base64 for API request
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    
    def test_single_image(self, image_path: str, generate_gradcam: bool = True) -> Dict[str, Any]:
        """
        Test fraud detection on a single image
        
        Args:
            image_path: Path to test image
            generate_gradcam: Whether to generate Grad-CAM explanation
            
        Returns:
            Complete test results including API response and Grad-CAM
        """
        print(f"\nTesting image: {Path(image_path).name}")
        print("-" * 40)
        
        # Encode image for API
        image_b64 = self.encode_image(image_path)
        
        # Create API request
        payload = {'image_base64': image_b64}
        
        # Time the API call
        start_time = time.time()
        
        try:
            # Call fraud detection API
            response = self.predictor.predict(payload)
            api_time = time.time() - start_time
            
            # Parse response if it's a string
            if isinstance(response, str):
                response = json.loads(response)
            
            # Display API results
            print(f"📊 API Response ({api_time:.2f}s):")
            print(f"   Prediction: {response['predicted_label']}")
            print(f"   Confidence: {response['confidence']:.3f}")
            print(f"   Risk Score: {response['risk_score']:.3f}")
            print(f"   Fraudulent: {response['is_fraudulent']}")
            print(f"   Interpretation: {response['interpretation']}")
            
            # Generate Grad-CAM explanation if requested
            gradcam_result = None
            if generate_gradcam:
                gradcam_result = self._generate_gradcam_explanation(
                    image_path, 
                    response['predicted_class']
                )
            
            return {
                'image_path': image_path,
                'api_response': response,
                'api_time': api_time,
                'gradcam_result': gradcam_result,
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'api_time': time.time() - start_time
            }
    
    def _generate_gradcam_explanation(self, image_path: str, predicted_class: int) -> Dict[str, Any]:
        """
        Generate Grad-CAM explanation for the prediction
        
        Args:
            image_path: Path to image
            predicted_class: Class predicted by the model
            
        Returns:
            Grad-CAM explanation results
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            
            # Generate Grad-CAM
            targets = [ClassifierOutputTarget(predicted_class)]
            input_tensor = image_tensor.unsqueeze(0)
            
            grayscale_cam = self.gradcam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Create visualization
            original_image = self._tensor_to_numpy(image_tensor)
            cam_overlay = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
            
            # Save visualizations
            image_name = Path(image_path).stem
            
            # Save original
            original_save_path = f"demo_original_{image_name}.jpg"
            Image.fromarray((original_image * 255).astype(np.uint8)).save(original_save_path)
            
            # Save heatmap
            heatmap_save_path = f"demo_heatmap_{image_name}.jpg"
            plt.figure(figsize=(6, 6))
            plt.imshow(grayscale_cam, cmap='jet')
            plt.axis('off')
            plt.title(f'Grad-CAM Heatmap - {["Authentic", "Tampered"][predicted_class]}')
            plt.savefig(heatmap_save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            # Save overlay
            overlay_save_path = f"demo_overlay_{image_name}.jpg"
            Image.fromarray(cam_overlay).save(overlay_save_path)
            
            # Calculate explanation metrics
            high_activation_ratio = np.sum(grayscale_cam > 0.7) / grayscale_cam.size
            avg_intensity = np.mean(grayscale_cam)
            
            print(f"🔍 Grad-CAM Analysis:")
            print(f"   Average intensity: {avg_intensity:.3f}")
            print(f"   High activation regions: {high_activation_ratio:.1%}")
            print(f"   Saved visualizations:")
            print(f"     - Original: {original_save_path}")
            print(f"     - Heatmap: {heatmap_save_path}")
            print(f"     - Overlay: {overlay_save_path}")
            
            return {
                'avg_intensity': float(avg_intensity),
                'high_activation_ratio': float(high_activation_ratio),
                'max_activation': float(np.max(grayscale_cam)),
                'original_image_path': original_save_path,
                'heatmap_path': heatmap_save_path,
                'overlay_path': overlay_save_path,
                'explanation_available': True
            }
            
        except Exception as e:
            print(f"⚠️  Grad-CAM generation failed: {e}")
            return {
                'explanation_available': False,
                'error': str(e)
            }
    
    def _tensor_to_numpy(self, tensor):
        """Convert normalized tensor to numpy array for visualization"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        denorm_tensor = tensor * std + mean
        denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
        
        # Convert to numpy
        image_np = denorm_tensor.permute(1, 2, 0).numpy()
        return image_np.astype(np.float32)
    
    def run_batch_test(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run batch test on multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of test results
        """
        print(f"\n{'='*60}")
        print(f"BATCH FRAUD DETECTION TEST - {len(image_paths)} IMAGES")
        print(f"{'='*60}")
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] Processing: {Path(image_path).name}")
            
            result = self.test_single_image(image_path, generate_gradcam=True)
            results.append(result)
            
            # Small delay between requests
            time.sleep(0.5)
        
        # Print summary
        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: List[Dict[str, Any]]):
        """Print summary of batch test results"""
        successful_results = [r for r in results if 'error' not in r]
        
        print(f"\n{'='*60}")
        print("BATCH TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total images tested: {len(results)}")
        print(f"Successful predictions: {len(successful_results)}")
        print(f"Failed predictions: {len(results) - len(successful_results)}")
        
        if successful_results:
            # API performance metrics
            api_times = [r['api_time'] for r in successful_results]
            print(f"\nAPI Performance:")
            print(f"  Average response time: {np.mean(api_times):.2f}s")
            print(f"  Min response time: {np.min(api_times):.2f}s")
            print(f"  Max response time: {np.max(api_times):.2f}s")
            
            # Prediction breakdown
            predictions = [r['api_response']['predicted_label'] for r in successful_results]
            authentic_count = predictions.count('Authentic')
            tampered_count = predictions.count('Tampered')
            
            print(f"\nPrediction Breakdown:")
            print(f"  Predicted as Authentic: {authentic_count}")
            print(f"  Predicted as Tampered: {tampered_count}")
            
            # Confidence analysis
            confidences = [r['api_response']['confidence'] for r in successful_results]
            print(f"\nConfidence Analysis:")
            print(f"  Average confidence: {np.mean(confidences):.3f}")
            print(f"  High confidence (>0.8): {sum(1 for c in confidences if c > 0.8)}")
            print(f"  Low confidence (<0.6): {sum(1 for c in confidences if c < 0.6)}")


def run_complete_demo():
    """
    Run complete fraud detection demonstration
    """
    print("=" * 60)
    print("FRAUD DETECTION SYSTEM - COMPLETE DEMONSTRATION")
    print("=" * 60)
    
    # Configuration
    model_path = 'fraud_detection_complete_20250603_145722.pth'
    
    # Check prerequisites
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Find test images
    test_images = []
    for pattern in ['test_authentic_*.jpg', 'test_tampered_*.jpg']:
        test_images.extend(list(Path('.').glob(pattern)))
    
    if not test_images:
        print("❌ No test images found. Please ensure test images are in the current directory.")
        return
    
    print(f"Found {len(test_images)} test images")
    
    try:
        # Step 1: Deploy local endpoint
        print("\n🚀 Deploying SageMaker local endpoint...")
        predictor = deploy_fraud_detection_model(model_path)
        
        # Step 2: Initialize tester
        print("\n🔧 Initializing API tester with Grad-CAM...")
        tester = FraudDetectionAPITester(predictor, model_path)
        
        # Step 3: Run batch test
        print("\n🧪 Running fraud detection tests...")
        test_results = tester.run_batch_test([str(img) for img in test_images])
        
        # Step 4: Summary
        print(f"\n✅ Demonstration completed successfully!")
        print(f"📁 Check current directory for generated Grad-CAM visualizations")
        print(f"📊 {len(test_results)} images processed")
        
        return test_results
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        return None


if __name__ == "__main__":
    # Run the complete demonstration
    results = run_complete_demo()
    
    if results:
        print(f"\n🎉 Fraud detection system demonstration complete!")
        print(f"The system is ready for assessment presentation.")
    else:
        print(f"\n💥 Demonstration failed. Please check the error messages above.")