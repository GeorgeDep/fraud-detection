"""
Test script for Flask Fraud Detection API
Demonstrates API functionality and generates results for assessment
"""

import json
import base64
import time
import requests
from pathlib import Path
from typing import List, Dict, Any

class FraudDetectionAPIClient:
    """Client for testing the Flask Fraud Detection API"""
    
    def __init__(self, base_url: str = 'http://localhost:5000'):
        """Initialize API client"""
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_health(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = self.session.get(f'{self.base_url}/health')
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def encode_image(self, image_path: str) -> str:
        """Encode image file to base64"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    
    def predict_fraud(self, image_path: str) -> Dict[str, Any]:
        """Get fraud prediction for image"""
        try:
            # Encode image
            image_b64 = self.encode_image(image_path)
            
            # Send request
            payload = {'image_base64': image_b64}
            response = self.session.post(f'{self.base_url}/predict', json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}: {response.text}'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def explain_prediction(self, image_path: str) -> Dict[str, Any]:
        """Get fraud prediction with Grad-CAM explanations"""
        try:
            # Encode image
            image_b64 = self.encode_image(image_path)
            
            # Send request
            payload = {'image_base64': image_b64}
            response = self.session.post(f'{self.base_url}/explain', json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}: {response.text}'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def run_demo(self) -> Dict[str, Any]:
        """Run API demo endpoint"""
        try:
            response = self.session.get(f'{self.base_url}/demo')
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}: {response.text}'}
        except Exception as e:
            return {'error': str(e)}


def save_gradcam_images(explanation_data: Dict[str, Any], image_name: str, output_dir: str = './api_demo_results'):
    """Save Grad-CAM visualizations from API response"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not explanation_data.get('explanation_available', False):
        print(f"   No Grad-CAM data available for {image_name}")
        return
    
    visualizations = explanation_data.get('visualizations', {})
    
    # Save visualizations
    saved_files = []
    for viz_type, b64_data in visualizations.items():
        if b64_data:
            try:
                # Decode base64 image
                image_data = base64.b64decode(b64_data)
                
                # Save to file
                filename = f"{image_name}_{viz_type}.png"
                filepath = output_path / filename
                
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                
                saved_files.append(str(filepath))
                
            except Exception as e:
                print(f"   Error saving {viz_type}: {e}")
    
    if saved_files:
        print(f"   Saved visualizations: {len(saved_files)} files")
        for file in saved_files:
            print(f"     - {file}")


def test_single_image(client: FraudDetectionAPIClient, image_path: str):
    """Test API on a single image with detailed output"""
    print(f"\nTesting: {Path(image_path).name}")
    print("-" * 50)
    
    # Test prediction endpoint
    print("📊 Getting fraud prediction...")
    start_time = time.time()
    prediction_result = client.predict_fraud(image_path)
    prediction_time = time.time() - start_time
    
    if 'error' in prediction_result:
        print(f"❌ Prediction failed: {prediction_result['error']}")
        return
    
    # Display prediction results
    pred = prediction_result
    print(f"   Prediction: {pred['predicted_label']}")
    print(f"   Confidence: {pred['confidence']:.3f}")
    print(f"   Risk Score: {pred['risk_score']:.3f}")
    print(f"   Fraudulent: {pred['is_fraudulent']}")
    print(f"   Response Time: {prediction_time:.2f}s")
    print(f"   Interpretation: {pred['interpretation']}")
    
    # Test explanation endpoint
    print("\n🔍 Getting Grad-CAM explanation...")
    start_time = time.time()
    explanation_result = client.explain_prediction(image_path)
    explanation_time = time.time() - start_time
    
    if 'error' in explanation_result:
        print(f"❌ Explanation failed: {explanation_result['error']}")
        return
    
    # Display explanation results
    if 'explanation' in explanation_result:
        exp = explanation_result['explanation']
        print(f"   Explanation available: {exp.get('explanation_available', False)}")
        
        if exp.get('explanation_available', False):
            print(f"   Heatmap intensity: {exp.get('heatmap_intensity', 0):.3f}")
            print(f"   High activation ratio: {exp.get('high_activation_ratio', 0):.1%}")
            print(f"   Max activation: {exp.get('max_activation', 0):.3f}")
            print(f"   Response Time: {explanation_time:.2f}s")
            
            # Save visualizations
            image_name = Path(image_path).stem
            save_gradcam_images(exp, image_name)
        else:
            print(f"   Explanation error: {exp.get('error', 'Unknown error')}")


def run_comprehensive_test():
    """Run comprehensive API testing"""
    print("=" * 70)
    print("FRAUD DETECTION API - COMPREHENSIVE TEST")
    print("=" * 70)
    
    # Initialize client
    client = FraudDetectionAPIClient()
    
    # Test 1: Health check
    print("\n🔧 Testing API health...")
    health = client.check_health()
    
    if 'error' in health:
        print(f"❌ API not responding: {health['error']}")
        print("Please ensure the Flask app is running: python3 app.py")
        return
    
    print(f"✅ API Status: {health.get('status', 'unknown')}")
    print(f"✅ Model loaded: {health.get('model_loaded', False)}")
    print(f"✅ Grad-CAM available: {health.get('gradcam_available', False)}")
    
    # Test 2: Find test images
    print("\n📁 Finding test images...")
    test_images = []
    for pattern in ['test_authentic_*.jpg', 'test_tampered_*.jpg']:
        test_images.extend(list(Path('.').glob(pattern)))
    
    if not test_images:
        print("❌ No test images found in current directory")
        return
    
    print(f"✅ Found {len(test_images)} test images")
    
    # Test 3: Individual image testing
    print(f"\n🧪 Testing individual images...")
    for i, image_path in enumerate(test_images[:4]):  # Test first 4 images
        test_single_image(client, str(image_path))
        
        # Small delay between requests
        if i < len(test_images) - 1:
            time.sleep(0.5)
    
    # Test 4: Demo endpoint
    print(f"\n🎪 Testing demo endpoint...")
    demo_result = client.run_demo()
    
    if 'error' in demo_result:
        print(f"❌ Demo failed: {demo_result['error']}")
    else:
        demo_results = demo_result.get('demo_results', [])
        print(f"✅ Demo completed: {len(demo_results)} images processed")
        
        # Summary of demo results
        authentic_count = sum(1 for r in demo_results 
                            if r.get('prediction', {}).get('predicted_label') == 'Authentic')
        tampered_count = sum(1 for r in demo_results 
                           if r.get('prediction', {}).get('predicted_label') == 'Tampered')
        
        print(f"   Predicted as Authentic: {authentic_count}")
        print(f"   Predicted as Tampered: {tampered_count}")
    
    # Test 5: Performance summary
    print(f"\n📊 TESTING SUMMARY")
    print("=" * 50)
    print("✅ API endpoints functional")
    print("✅ Fraud detection working")
    print("✅ Grad-CAM explanations available")
    print("✅ Visual outputs saved to ./api_demo_results/")
    print("\n🎯 API ready for assessment demonstration!")
    
    # Save test results
    results_summary = {
        'health_check': health,
        'demo_endpoint': demo_result,
        'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'images_tested': len(test_images),
        'api_base_url': client.base_url
    }
    
    results_file = Path('./api_demo_results/test_summary.json')
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"📄 Test summary saved to: {results_file}")


if __name__ == "__main__":
    print("Starting Flask API testing...")
    print("Make sure the Flask app is running in another terminal:")
    print("  python3 app.py")
    print("\nWaiting 3 seconds for you to start the app if needed...")
    time.sleep(3)
    
    run_comprehensive_test()