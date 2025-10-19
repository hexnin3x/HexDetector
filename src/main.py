"""
HexDetector Main Execution Pipeline

Main entry point for the HexDetector network traffic anomaly detection system.
Supports multiple execution modes: demo, full experiments, and custom configurations.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import time

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from data.data_loader import DataLoader
from data.preprocessing import preprocess_data
from features.build_features import build_features
from models.train import train_model
from models.evaluate import evaluate_model
from models.predict import make_predictions
from utils.logger import Logger
from utils.visualization import save_experiment_plots


class HexDetectorPipeline:
    """Main pipeline orchestrator for HexDetector"""
    
    def __init__(self, mode='demo', model_type='random_forest', samples=None, output_dir=None):
        """
        Initialize the HexDetector pipeline.
        
        Parameters:
        mode (str): Execution mode ('demo', 'full', 'custom')
        model_type (str): Type of ML model to use
        samples (int): Number of samples to use (None for all in full mode)
        output_dir (str): Directory to save results
        """
        self.logger = Logger('HexDetector.Pipeline')
        self.mode = mode
        self.model_type = model_type
        self.samples = samples
        self.output_dir = output_dir or settings.OUTPUT_DIR / f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.log_info(f"Initialized HexDetector Pipeline in {mode} mode")
        self.logger.log_info(f"Model type: {model_type}")
        self.logger.log_info(f"Output directory: {self.output_dir}")
    
    def run(self):
        """Execute the complete pipeline"""
        start_time = time.time()
        
        try:
            self.logger.log_step("HexDetector Pipeline Execution", 1)
            
            # Step 1: Load data
            data = self._load_data()
            if data is None or data.empty:
                self.logger.log_error("No data loaded. Exiting.")
                return None
            
            # Step 2: Preprocess data
            processed_data = self._preprocess_data(data)
            
            # Step 3: Build features
            features = self._build_features(processed_data)
            
            # Step 4: Train model
            model_results = self._train_model(features)
            
            # Step 5: Evaluate model
            evaluation = self._evaluate_model(model_results)
            
            # Step 6: Save results
            self._save_results(model_results, evaluation)
            
            # Calculate total time
            total_time = time.time() - start_time
            self.logger.log_time("Pipeline execution", total_time)
            
            self.logger.log_info("Pipeline completed successfully!")
            return model_results
            
        except Exception as e:
            self.logger.log_error(f"Pipeline execution failed: {str(e)}", e)
            return None
    
    def _load_data(self):
        """Load data based on execution mode"""
        self.logger.log_step("Loading Data", 2)
        
        loader = DataLoader()
        
        if self.mode == 'demo':
            nrows = settings.DEMO_SAMPLES_PER_FILE
            self.logger.log_info(f"Demo mode: Loading {nrows} samples per file")
        elif self.mode == 'custom' and self.samples:
            nrows = self.samples
            self.logger.log_info(f"Custom mode: Loading {nrows} samples")
        else:
            nrows = None
            self.logger.log_info("Full mode: Loading all available data")
        
        # Try to load from processed directory first
        processed_path = settings.DATA_PROCESSED_PATH
        if os.path.exists(processed_path) and os.listdir(processed_path):
            self.logger.log_info("Loading from processed data directory")
            data = loader.load_raw_data(str(processed_path), nrows=nrows)
        else:
            # Fall back to raw data
            raw_path = settings.DATA_RAW_PATH
            self.logger.log_info("Loading from raw data directory")
            data = loader.load_raw_data(str(raw_path), nrows=nrows)
        
        self.logger.log_info(f"Loaded data shape: {data.shape}")
        
        # Get data info
        info = loader.get_data_info(data)
        self.logger.log_info(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
        
        return data
    
    def _preprocess_data(self, data):
        """Preprocess the loaded data"""
        self.logger.log_step("Preprocessing Data", 3)
        
        processed = preprocess_data(data)
        self.logger.log_info(f"Processed data shape: {processed.shape}")
        
        # Save processed data
        processed_file = Path(self.output_dir) / 'processed_data.csv'
        processed.to_csv(processed_file, index=False)
        self.logger.log_info(f"Saved processed data to {processed_file}")
        
        return processed
    
    def _build_features(self, processed_data):
        """Build features from processed data"""
        self.logger.log_step("Building Features", 4)
        
        features = build_features(processed_data)
        self.logger.log_info(f"Built features shape: {features.shape}")
        
        # Save features
        features_file = Path(self.output_dir) / 'features.csv'
        features.to_csv(features_file, index=False)
        self.logger.log_info(f"Saved features to {features_file}")
        
        return features
    
    def _train_model(self, features):
        """Train the machine learning model"""
        self.logger.log_step("Training Model", 5)
        
        model_results = train_model(features, model_type=self.model_type)
        
        if model_results:
            self.logger.log_metrics({
                'accuracy': model_results.get('accuracy', 0),
                'precision': model_results.get('precision', 0),
                'recall': model_results.get('recall', 0),
                'f1': model_results.get('f1', 0)
            })
        
        return model_results
    
    def _evaluate_model(self, model_results):
        """Evaluate the trained model"""
        self.logger.log_step("Evaluating Model", 6)
        
        if not model_results:
            return None
        
        # Prepare test data
        test_data = pd.concat([
            model_results['X_test'],
            model_results['y_test']
        ], axis=1)
        
        evaluation = evaluate_model(
            model_results['model'],
            test_data,
            metrics=['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']
        )
        
        return evaluation
    
    def _save_results(self, model_results, evaluation):
        """Save all results and generate visualizations"""
        self.logger.log_step("Saving Results", 7)
        
        if not model_results:
            return
        
        # Save model
        import joblib
        model_file = Path(self.output_dir) / f'{self.model_type}_model.pkl'
        joblib.dump(model_results['model'], model_file)
        self.logger.log_info(f"Saved model to {model_file}")
        
        # Save metrics
        metrics_file = Path(self.output_dir) / 'metrics.csv'
        pd.DataFrame([{
            'model_type': self.model_type,
            'accuracy': model_results.get('accuracy', 0),
            'precision': model_results.get('precision', 0),
            'recall': model_results.get('recall', 0),
            'f1': model_results.get('f1', 0)
        }]).to_csv(metrics_file, index=False)
        self.logger.log_info(f"Saved metrics to {metrics_file}")
        
        # Save feature importance if available
        if model_results.get('feature_importance') is not None:
            importance_file = Path(self.output_dir) / 'feature_importance.csv'
            model_results['feature_importance'].to_csv(importance_file, index=False)
            self.logger.log_info(f"Saved feature importance to {importance_file}")
        
        # Generate visualizations
        try:
            save_experiment_plots(model_results, str(self.output_dir))
        except Exception as e:
            self.logger.log_warning(f"Could not generate visualizations: {str(e)}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='HexDetector: Network Traffic Anomaly Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with 10,000 samples
  python src/main.py --mode demo
  
  # Run full experiment with Random Forest
  python src/main.py --mode full --model random_forest
  
  # Run custom experiment with 50,000 samples
  python src/main.py --mode custom --samples 50000 --model xgboost
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='demo',
        choices=['demo', 'full', 'custom'],
        help='Execution mode (default: demo)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=list(settings.MODEL_TYPES.keys()),
        help='Model type to use (default: random_forest)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of samples to use in custom mode'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--attacks',
        nargs='+',
        default=None,
        help='Specific attack types to include'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize logger
    logger = Logger('HexDetector.Main')
    
    # Print header
    logger.log_separator('=', 70)
    logger.log_info("HexDetector: Network Traffic Anomaly Detection")
    logger.log_info(f"Version: 1.0.0")
    logger.log_info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log_separator('=', 70)
    
    # Log configuration
    logger.log_info(f"Mode: {args.mode}")
    logger.log_info(f"Model: {args.model}")
    if args.samples:
        logger.log_info(f"Samples: {args.samples}")
    if args.output:
        logger.log_info(f"Output: {args.output}")
    
    # Create and run pipeline
    pipeline = HexDetectorPipeline(
        mode=args.mode,
        model_type=args.model,
        samples=args.samples,
        output_dir=args.output
    )
    
    results = pipeline.run()
    
    if results:
        logger.log_info("Execution completed successfully!")
        logger.log_info(f"Results saved to: {pipeline.output_dir}")
        return 0
    else:
        logger.log_error("Execution failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())