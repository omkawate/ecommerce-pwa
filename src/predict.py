import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from data_loader import EuroSATDataLoader


class SatellitePredictor:
    """Handles loading trained model and making predictions."""
    
    def __init__(self, config_path="config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model_config = self.config['model']
        self.paths = self.config['paths']
        self.gui_config = self.config['gui']
        
        self.model = None
        self.class_names = self.config['dataset']['class_names']
        self.image_size = tuple(self.model_config['input_shape'][:2])
        
        self.data_loader = EuroSATDataLoader(config_path)
    
    def load_model(self, model_path=None):
        """Load trained model."""
        if model_path is None:
            model_path = self.paths['model_save_path']
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        print("Model loaded successfully!")
        return self.model
    
    def predict_single(self, image_path):
        """Predict class for a single image."""
        if self.model is None:
            self.load_model()
        
        img, original_img = self.data_loader.load_single_image(image_path)
        
        predictions = self.model.predict(img, verbose=0)
        pred_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_class_idx] * 100
        pred_class = self.class_names[pred_class_idx]
        
        all_probs = {self.class_names[i]: float(predictions[0][i] * 100) for i in range(len(self.class_names))}
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'all_probabilities': sorted_probs,
            'original_image': original_img,
            'processed_image': img[0]
        }
    
    def predict_batch(self, image_paths):
        """Predict classes for multiple images."""
        if self.model is None:
            self.load_model()
        
        results = []
        images = []
        
        for img_path in image_paths:
            try:
                img, original_img = self.data_loader.load_single_image(img_path)
                images.append(img[0])
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                results.append({'error': str(e), 'path': str(img_path)})
                continue
        
        if images:
            batch = np.array(images)
            predictions = self.model.predict(batch, verbose=0)
            
            for i, img_path in enumerate(image_paths):
                if i < len(predictions):
                    pred_class_idx = np.argmax(predictions[i])
                    confidence = predictions[i][pred_class_idx] * 100
                    pred_class = self.class_names[pred_class_idx]
                    
                    all_probs = {self.class_names[j]: float(predictions[i][j] * 100) 
                                for j in range(len(self.class_names))}
                    sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
                    
                    results.append({
                        'path': str(img_path),
                        'predicted_class': pred_class,
                        'confidence': confidence,
                        'all_probabilities': sorted_probs
                    })
        
        return results
    
    def predict_from_array(self, image_array):
        """Predict from a preprocessed image array."""
        if self.model is None:
            self.load_model()
        
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        predictions = self.model.predict(image_array, verbose=0)
        pred_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_class_idx] * 100
        pred_class = self.class_names[pred_class_idx]
        
        all_probs = {self.class_names[i]: float(predictions[0][i] * 100) for i in range(len(self.class_names))}
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'all_probabilities': sorted_probs
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Satellite Image Classification - Prediction')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, nargs='+', help='Paths to multiple images')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    
    args = parser.parse_args()
    
    predictor = SatellitePredictor(args.config)
    
    if args.model:
        predictor.load_model(args.model)
    else:
        predictor.load_model()
    
    if args.image:
        result = predictor.predict_single(args.image)
        print(f"\nPrediction: {result['predicted_class']} ({result['confidence']:.2f}%)")
        print("\nAll probabilities:")
        for cls, prob in result['all_probabilities'].items():
            print(f"  {cls}: {prob:.2f}%")
    
    elif args.batch:
        results = predictor.predict_batch(args.batch)
        print(f"\nBatch Prediction Results ({len(results)} images):")
        print("-" * 80)
        for r in results:
            if 'error' in r:
                print(f"{r['path']}: ERROR - {r['error']}")
            else:
                print(f"{r['path']}: {r['predicted_class']} ({r['confidence']:.2f}%)")
    
    else:
        print("Please provide --image or --batch argument")


if __name__ == "__main__":
    main()