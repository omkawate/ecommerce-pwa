import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import pandas as pd


def plot_training_history(history, save_dir):
    """Plot training and validation accuracy/loss curves."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    axes[0].plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'training_history.pdf', bbox_inches='tight')
    plt.close()
    
    if 'top3_acc' in history.history:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, history.history['top3_acc'], 'g-', label='Training Top-3 Acc', linewidth=2)
        ax.plot(epochs, history.history['val_top3_acc'], 'm-', label='Validation Top-3 Acc', linewidth=2)
        ax.set_title('Training and Validation Top-3 Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Top-3 Accuracy', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'top3_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_confusion_matrix(cm, class_names, save_dir, normalize=False):
    """Plot confusion matrix."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'confusion_matrix.pdf', bbox_inches='tight')
    plt.close()
    
    if not normalize:
        plot_confusion_matrix(cm, class_names, save_dir, normalize=True)


def save_classification_report(report, class_names, save_dir):
    """Save classification report as CSV and text."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(report).transpose()
    df.to_csv(save_dir / 'classification_report.csv', float_format='%.4f')
    
    with open(save_dir / 'classification_report.txt', 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1-score']:.4f}\n")
                f.write(f"  Support:   {int(metrics['support'])}\n\n")
        
        f.write("\nAVERAGES:\n")
        f.write("-" * 30 + "\n")
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report:
                metrics = report[avg_type]
                f.write(f"{avg_type}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1-score']:.4f}\n")
                f.write(f"  Support:   {int(metrics['support'])}\n\n")
        
        if 'accuracy' in report:
            f.write(f"Overall Accuracy: {report['accuracy']:.4f}\n")


def plot_sample_predictions(images, true_labels, pred_labels, pred_probs, class_names, save_dir, num_samples=16):
    """Plot sample predictions with confidence scores."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    for i in range(num_samples):
        ax = axes[i]
        img = images[i]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        ax.axis('off')
        
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        confidence = pred_probs[i][pred_labels[i]] * 100
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        title = f"True: {true_class}\nPred: {pred_class} ({confidence:.1f}%)"
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_distribution(labels, class_names, save_dir, title="Class Distribution"):
    """Plot class distribution in dataset."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([class_names[i] for i in unique], counts, color='steelblue', edgecolor='black')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                 str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_confidence(pred_probs, class_names, save_dir):
    """Plot prediction confidence distribution."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    max_probs = np.max(pred_probs, axis=1)
    pred_classes = np.argmax(pred_probs, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(max_probs, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Confidence', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].axvline(np.mean(max_probs), color='red', linestyle='--', label=f'Mean: {np.mean(max_probs):.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    class_counts = np.bincount(pred_classes, minlength=len(class_names))
    axes[1].bar(range(len(class_names)), class_counts, color='coral', edgecolor='black')
    axes[1].set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_model_architecture_diagram(model, save_dir):
    """Create a visual representation of model architecture."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file=str(save_dir / 'model_architecture.png'),
                   show_shapes=True, show_layer_names=True, rankdir='TB', dpi=150)
        print(f"Model architecture diagram saved to {save_dir / 'model_architecture.png'}")
    except Exception as e:
        print(f"Could not create model architecture diagram: {e}")