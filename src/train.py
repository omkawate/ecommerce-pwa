import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import sys

sys.path.append(str(Path(__file__).parent))
from data_loader import get_data_loader
from model import create_cnn_model, get_callbacks, save_model_architecture, count_parameters, get_model_summary
from utils.visualization import plot_training_history, plot_confusion_matrix, save_classification_report


class SatelliteTrainer:
    """Handles model training, evaluation, and visualization."""
    
    def __init__(self, config_path="config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model_config = self.config['model']
        self.paths = self.config['paths']
        self.class_names = self.config['dataset']['class_names']
        
        self.model = None
        self.history = None
        self.data = None
    
    def prepare_data(self):
        """Load and prepare dataset."""
        print("=" * 60)
        print("PREPARING DATA")
        print("=" * 60)
        
        use_synthetic = self.config['dataset'].get('use_synthetic', False)
        data_loader = get_data_loader("config.json")
        self.data = data_loader.prepare_data(use_synthetic=use_synthetic)
        self.class_names = self.data['class_names']
        
        print(f"Classes: {self.class_names}")
        print(f"Train samples: {len(self.data['train'][0])}")
        print(f"Val samples: {len(self.data['val'][0])}")
        print(f"Test samples: {len(self.data['test'][0])}")
        
        if use_synthetic:
            print("\n*** USING SYNTHETIC DATA FOR TESTING ***")
            print("Set 'use_synthetic': false in config.json to use real EuroSAT data")
    
    def build_model(self):
        """Build and compile the CNN model."""
        print("\n" + "=" * 60)
        print("BUILDING MODEL")
        print("=" * 60)
        
        input_shape = tuple(self.model_config['input_shape'])
        num_classes = self.model_config['num_classes']
        learning_rate = self.model_config['learning_rate']
        
        self.model = create_cnn_model(input_shape, num_classes, learning_rate)
        
        total, trainable, non_trainable = count_parameters(self.model)
        print(f"\nModel Summary:")
        print(get_model_summary(self.model))
        print(f"\nTotal parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Non-trainable parameters: {non_trainable:,}")
        
        save_model_architecture(self.model, self.paths['model_json_path'])
    
    def train(self):
        """Train the model."""
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        
        X_train, y_train, train_gen = self.data['train']
        X_val, y_val, val_gen = self.data['val']
        
        epochs = self.model_config['epochs']
        batch_size = self.model_config['batch_size']
        
        steps_per_epoch = len(X_train) // batch_size
        validation_steps = len(X_val) // batch_size
        
        callbacks = get_callbacks(self.config, Path(self.paths['model_save_path']))
        
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        start_time = time.time()
        
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        self.model.save(self.paths['model_save_path'])
        print(f"Model saved to {self.paths['model_save_path']}")
    
    def evaluate(self):
        """Evaluate model on test set."""
        print("\n" + "=" * 60)
        print("EVALUATING MODEL")
        print("=" * 60)
        
        X_test, y_test, test_gen = self.data['test']
        
        test_steps = len(X_test) // self.model_config['batch_size']
        if len(X_test) % self.model_config['batch_size'] != 0:
            test_steps += 1
        
        test_loss, test_acc, test_top3 = self.model.evaluate(
            test_gen, steps=test_steps, verbose=1
        )
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Test Top-3 Accuracy: {test_top3:.4f} ({test_top3*100:.2f}%)")
        
        y_pred_probs = self.model.predict(test_gen, steps=test_steps, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test[:len(y_pred)], axis=1)
        
        return y_true, y_pred, y_pred_probs, test_loss, test_acc
    
    def generate_reports(self, y_true, y_pred, y_pred_probs):
        """Generate detailed evaluation reports."""
        print("\n" + "=" * 60)
        print("GENERATING REPORTS")
        print("=" * 60)
        
        plots_dir = Path(self.paths['plots_dir'])
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_training_history(self.history, plots_dir)
        
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, self.class_names, plots_dir)
        
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        save_classification_report(report, self.class_names, plots_dir)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        print("\nPer-class Metrics:")
        print("-" * 80)
        print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 80)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<25} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")
        print("-" * 80)
        
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        print(f"\nMacro Avg:     Precision={macro_precision:.4f} Recall={macro_recall:.4f} F1={macro_f1:.4f}")
        print(f"Weighted Avg:  Precision={weighted_precision:.4f} Recall={weighted_recall:.4f} F1={weighted_f1:.4f}")
        print(f"Overall Acc:   {accuracy_score(y_true, y_pred):.4f}")
        
        return {
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'per_class': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'support': support.tolist()
            },
            'macro_avg': {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1},
            'weighted_avg': {'precision': weighted_precision, 'recall': weighted_recall, 'f1': weighted_f1},
            'accuracy': float(accuracy_score(y_true, y_pred))
        }
    
    def run_complete_pipeline(self):
        """Run the complete training and evaluation pipeline."""
        try:
            self.prepare_data()
            self.build_model()
            self.train()
            y_true, y_pred, y_pred_probs, test_loss, test_acc = self.evaluate()
            metrics = self.generate_reports(y_true, y_pred, y_pred_probs)
            
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE!")
            print("=" * 60)
            print(f"Final Test Accuracy: {test_acc*100:.2f}%")
            print(f"Model saved to: {self.paths['model_save_path']}")
            print(f"Plots saved to: {self.paths['plots_dir']}")
            
            return metrics
            
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    trainer = SatelliteTrainer("config.json")
    metrics = trainer.run_complete_pipeline()
    
    if metrics:
        print("\nTraining completed successfully!")
    else:
        print("\nTraining failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()