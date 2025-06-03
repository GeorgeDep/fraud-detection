"""
Deploy fraud detection model using SageMaker local mode
Creates a local endpoint for testing and demonstration
"""

import os
import tarfile
import shutil
from pathlib import Path
import sagemaker
from sagemaker.pytorch import PyTorchModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalFraudDetectionDeployer:
    """
    Handles local deployment of fraud detection model using SageMaker
    """
    
    def __init__(self, model_path: str):
        """
        Initialize deployer with model path
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = Path(model_path)
        self.work_dir = Path('./model_artifacts')
        self.session = sagemaker.LocalSession()
        self.role = 'arn:aws:iam::123456789012:role/MySageMakerRole'  # Dummy role for local mode
        
        logger.info(f"Initialized local deployer for model: {self.model_path}")
    
    def create_model_package(self):
        """
        Create model.tar.gz package for SageMaker deployment
        
        Returns:
            Path to created model package
        """
        logger.info("Creating model package...")
        
        # Create working directory
        self.work_dir.mkdir(exist_ok=True)
        
        # Copy model file
        model_dest = self.work_dir / self.model_path.name
        shutil.copy2(self.model_path, model_dest)
        logger.info(f"Copied model to {model_dest}")
        
        # Copy inference script
        inference_src = Path('./inference.py')
        if not inference_src.exists():
            raise FileNotFoundError("inference.py not found. Please ensure it's in the current directory.")
        
        inference_dest = self.work_dir / 'inference.py'
        shutil.copy2(inference_src, inference_dest)
        logger.info("Copied inference script")
        
        # Create model package
        package_path = Path('./model.tar.gz')
        
        with tarfile.open(package_path, 'w:gz') as tar:
            tar.add(model_dest, arcname=model_dest.name)
            tar.add(inference_dest, arcname='inference.py')
        
        logger.info(f"Model package created: {package_path}")
        return package_path
    
    def deploy_local_endpoint(self, package_path: Path, endpoint_name: str = 'fraud-detection-local'):
        """
        Deploy model to local SageMaker endpoint
        
        Args:
            package_path: Path to model package
            endpoint_name: Name for the endpoint
            
        Returns:
            SageMaker predictor instance
        """
        logger.info(f"Deploying local endpoint: {endpoint_name}")
        
        # Create PyTorch model
        pytorch_model = PyTorchModel(
            model_data=f"file://{package_path.absolute()}",
            role=self.role,
            entry_point='inference.py',
            framework_version='1.9.0',
            py_version='py38',
            sagemaker_session=self.session
        )
        
        logger.info("Created PyTorch model configuration")
        
        # Deploy to local endpoint
        logger.info("Starting local deployment (this may take a few minutes)...")
        predictor = pytorch_model.deploy(
            initial_instance_count=1,
            instance_type='local',  # This runs locally
            endpoint_name=endpoint_name
        )
        
        logger.info(f"Local endpoint deployed successfully: {endpoint_name}")
        return predictor
    
    def complete_deployment(self):
        """
        Run complete deployment pipeline
        
        Returns:
            Deployed predictor instance
        """
        try:
            logger.info("Starting complete local deployment...")
            
            # Step 1: Create model package
            package_path = self.create_model_package()
            
            # Step 2: Deploy local endpoint
            predictor = self.deploy_local_endpoint(package_path)
            
            logger.info("Local deployment completed successfully!")
            logger.info(f"Endpoint ready for testing")
            
            return predictor
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise
        
        finally:
            # Clean up temporary files
            if self.work_dir.exists():
                shutil.rmtree(self.work_dir)
                logger.info("Cleaned up temporary files")
    
    def test_endpoint_health(self, predictor):
        """
        Test if the endpoint is responding correctly
        
        Args:
            predictor: Deployed predictor instance
            
        Returns:
            Boolean indicating if endpoint is healthy
        """
        try:
            # Create a simple test payload
            import base64
            from PIL import Image
            import io
            
            # Create a small test image
            test_image = Image.new('RGB', (224, 224), color='white')
            buffer = io.BytesIO()
            test_image.save(buffer, format='JPEG')
            test_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            test_payload = {
                'image_base64': test_image_b64
            }
            
            # Send test request
            response = predictor.predict(test_payload)
            logger.info("Endpoint health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Endpoint health check failed: {e}")
            return False


def deploy_fraud_detection_model(model_path: str):
    """
    Main deployment function
    
    Args:
        model_path: Path to the trained model file
        
    Returns:
        Deployed predictor instance
    """
    print("=" * 60)
    print("FRAUD DETECTION MODEL - LOCAL SAGEMAKER DEPLOYMENT")
    print("=" * 60)
    
    # Check prerequisites
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not Path('./inference.py').exists():
        raise FileNotFoundError("inference.py not found in current directory")
    
    print(f"Model file: {model_path}")
    print(f"Deployment mode: SageMaker Local")
    print(f"Target: Local endpoint for testing")
    
    # Initialize deployer
    deployer = LocalFraudDetectionDeployer(model_path)
    
    # Run deployment
    predictor = deployer.complete_deployment()
    
    # Test endpoint health
    if deployer.test_endpoint_health(predictor):
        print("\n✅ Deployment successful!")
        print(f"✅ Endpoint is healthy and ready for use")
        print(f"✅ You can now test fraud detection using the predictor")
    else:
        print("\n⚠️  Deployment completed but endpoint health check failed")
    
    return predictor


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python deploy_local.py <model_path>")
        print("Example: python deploy_local.py fraud_detection_complete_20250603_145722.pth")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    try:
        predictor = deploy_fraud_detection_model(model_path)
        print(f"\nDeployment complete!")
        print(f"Use the returned predictor object to make predictions")
        print(f"Run test_demo.py to see the endpoint in action")
        
    except Exception as e:
        print(f"Deployment failed: {e}")
        sys.exit(1)