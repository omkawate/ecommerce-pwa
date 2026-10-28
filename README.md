# TerraScope — Satellite Spatial Intelligence

Professional browser workspace using the original pretrained EuroSAT CNN.

**Start here:** [Setup and presentation walkthrough](PRESENTATION.md).

```sh
python -m pip install -r requirements-spatial.txt
python -m streamlit run spatial_app.py
```

Includes image inspection, viewport zoom, batch inference, confidence distributions, uncertainty flags, session history, JSON/CSV downloads, and original training evidence. Model and training configuration remain unchanged; see MODEL_INTEGRITY.json.

## Original project documentation

# 🛰 Satellite Image Detection and Classification System

A complete **CNN-based satellite image classification system** built with Python, TensorFlow/Keras for detecting and classifying land use/land cover categories from satellite imagery.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [CNN Architecture](#cnn-architecture)
- [Usage](#usage)
- [Training](#training)
- [Prediction](#prediction)
- [GUI Application](#gui-application)
- [Results & Visualization](#results--visualization)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- **🎯 10-Class Classification**: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- **🧠 Deep CNN Architecture**: 4 convolutional blocks with BatchNorm, Dropout, and Global Average Pooling
- **📊 Complete Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Top-3 Accuracy
- **📈 Training Visualization**: Loss/Accuracy curves, Confusion Matrix heatmaps, Class distribution
- **🖥️ User-Friendly GUI**: Drag-and-drop interface for single/batch image classification
- **⚡ Data Augmentation**: Rotation, flip, zoom, shear for robust training
- **💾 Model Persistence**: Save/load trained models (HDF5 + JSON architecture)
- **🔧 Configurable**: JSON-based configuration for all hyperparameters
- **🛡️ Error Handling**: Graceful handling of missing files, wrong formats, etc.

---

## 📁 Project Structure

```
SATELLITE IMAGE AND DETECTION/
├── config.json                 # Configuration file
├── requirements.txt            # Python dependencies
├── main.py                     # Main entry point
├── README.md                   # This file
├── data/
│   └── EuroSAT/               # Dataset (auto-downloaded)
├── models/
│   ├── satellite_cnn_model.h5 # Trained model weights
│   └── model_architecture.json # Model architecture
├── outputs/
│   ├── plots/                 # Training plots & evaluation charts
│   └── predictions/           # Prediction results
├── logs/                      # TensorBoard logs
└── src/
    ├── __init__.py
    ├── data_loader.py         # Dataset loading & preprocessing
    ├── model.py               # CNN model definitions
    ├── train.py               # Training pipeline
    ├── predict.py             # Inference/prediction
    ├── utils/
    │   ├── __init__.py
    │   └── visualization.py   # Plotting utilities
    └── gui/
        ├── __init__.py
        └── app.py             # Tkinter GUI application
```

---

## 🚀 Installation

### Prerequisites
- **Python 3.8+** (recommended: 3.10 or 3.11)
- **Windows 10/11** (tested on Windows)
- **Git** (optional, for cloning)

### Quick Setup

```bash
# 1. Navigate to project directory
cd "C:\Users\omkaw\OneDrive\Desktop\health\SATELLITE IMAGE AND DETECTION"

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

### Requirements
```
tensorflow>=2.13.0
numpy>=1.24.0
pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
tqdm>=4.65.0
requests>=2.31.0
```

---

## 📊 Dataset

### EuroSAT Dataset
- **Source**: [EuroSAT](https://github.com/phelber/eurosat) - Sentinel-2 satellite imagery
- **Classes**: 10 land use/land cover categories
- **Images**: 27,000 labeled images (64×64 pixels, RGB)
- **Split**: Automatic train/validation/test split (70%/20%/10%)

### Class Distribution
| Class | Description |
|-------|-------------|
| AnnualCrop | Annual crop fields |
| Forest | Forested areas |
| HerbaceousVegetation | Grasslands, meadows |
| Highway | Highways and major roads |
| Industrial | Industrial buildings/areas |
| Pasture | Pasture lands |
| PermanentCrop | Permanent crops (orchards, vineyards) |
| Residential | Residential buildings/areas |
| River | Rivers and waterways |
| SeaLake | Lakes and sea water |

### Auto-Download
The dataset is **automatically downloaded** on first run (~2 GB). No manual download needed!

---

## 🧠 CNN Architecture

### Model: `SatelliteCNN`

```
Input: 64×64×3 RGB Image
│
├─ Block 1: Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
├─ Block 2: Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
├─ Block 3: Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.3)
├─ Block 4: Conv2D(256, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.3)
│
├─ GlobalAveragePooling2D
│
├─ Dense(512) → BatchNorm → ReLU → Dropout(0.5)
├─ Dense(256) → BatchNorm → ReLU → Dropout(0.5)
│
└─ Output: Dense(10, Softmax)
```

### Key Design Choices
| Component | Purpose |
|-----------|---------|
| **Batch Normalization** | Faster convergence, better generalization |
| **L2 Regularization (1e-4)** | Prevent overfitting |
| **Global Average Pooling** | Reduce parameters, spatial invariance |
| **Progressive Dropout** | Increasing dropout in deeper layers |
| **Adam Optimizer** | Adaptive learning rates |
| **ReduceLROnPlateau** | Automatic LR scheduling |

### Model Statistics
- **Total Parameters**: ~2.1M
- **Trainable Parameters**: ~2.1M
- **Model Size**: ~8 MB (HDF5)

---

## 💻 Usage

### Main Commands

```bash
# Show help
python main.py help

# Train model
python main.py train

# Predict single image
python main.py predict --image path/to/image.jpg

# Predict multiple images
python main.py predict --batch img1.jpg img2.jpg img3.jpg

# Launch GUI
python main.py gui
```

---

## 🏋️ Training

### Basic Training
```bash
python main.py train
```

### Custom Training Parameters
```bash
# Modify config.json or pass arguments
# Edit config.json for:
# - epochs (default: 50)
# - batch_size (default: 32)
# - learning_rate (default: 0.001)
# - validation_split (default: 0.2)
```

### Training Process
1. **Auto-download** EuroSAT dataset (~2 GB)
2. **Preprocess**: Resize to 64×64, normalize to [0,1]
3. **Split**: 70% train / 20% validation / 10% test
4. **Augment**: Random rotation, flip, zoom, shear
5. **Train**: With EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
6. **Evaluate**: On held-out test set
7. **Visualize**: Generate all plots and reports

### Expected Training Time
| Hardware | Time (50 epochs) |
|----------|-----------------|
| CPU (modern) | 30-60 minutes |
| GPU (RTX 3060+) | 3-5 minutes |
| Google Colab (T4) | 4-6 minutes |

### Training Outputs
Saved to `outputs/plots/`:
- `training_history.png` - Loss/Accuracy curves
- `top3_accuracy.png` - Top-3 accuracy curves
- `confusion_matrix.png` - Confusion matrix heatmap
- `confusion_matrix_normalized.png` - Normalized confusion matrix
- `classification_report.csv/txt` - Detailed metrics
- `class_distribution.png` - Dataset class balance

---

## 🔮 Prediction

### Command Line
```bash
# Single image
python main.py predict --image satellite_image.jpg

# Multiple images
python main.py predict --batch image1.jpg image2.jpg image3.jpg

# Custom model
python main.py predict --image test.jpg --model models/best_model.h5
```

### Output Format
```
Prediction: Residential (94.32%)

All probabilities:
  Residential:          94.32%
  Industrial:            3.12%
  AnnualCrop:            1.05%
  Highway:               0.89%
  ...
```

### Python API
```python
from src.predict import SatellitePredictor

predictor = SatellitePredictor()
predictor.load_model()

# Single prediction
result = predictor.predict_single("image.jpg")
print(f"{result['predicted_class']} ({result['confidence']:.2f}%)")

# Batch prediction
results = predictor.predict_batch(["img1.jpg", "img2.jpg"])
for r in results:
    print(f"{r['path']}: {r['predicted_class']} ({r['confidence']:.2f}%)")
```

---

## 🖥️ GUI Application

### Launch
```bash
python main.py gui
```

### Features
- **📁 Upload Image**: Click "Upload Image" or drag & drop
- **🔍 Classify**: Instant prediction with confidence score
- **📊 Probability Bars**: Visual breakdown of all 10 classes
- **📂 Batch Process**: Select folder → process all images
- **💾 Export**: Save results to CSV
- **🎨 Clean Interface**: Modern, responsive design

### GUI Screenshots
```
┌─────────────────────────────────────────────────────────┐
│  🛰 Satellite Image Classification                      │
│  Upload a satellite image to classify land use/cover   │
├─────────────────────┬───────────────────────────────────┤
│                     │  Prediction Results               │
│   [IMAGE PREVIEW]   │  ─────────────────                │
│                     │  Predicted Class: Residential     │
│                     │  Confidence: 94.32%               │
│  [Upload] [Classify]│  ─────────────────                │
│  [Clear]            │  All Class Probabilities:         │
│                     │  ████████████ Residential 94.3%  │
│                     │  ███ Industrial       3.1%       │
│                     │  ...                              │
├─────────────────────┴───────────────────────────────────┤
│  Batch Processing: [Select Folder] [Export Results]     │
│  Status: Ready  |  Model: Loaded ✓                      │
└─────────────────────────────────────────────────────────┘
```

---

## 📈 Results & Visualization

### Sample Results (Expected)
| Metric | Value |
|--------|-------|
| Test Accuracy | 95-98% |
| Macro F1-Score | 94-97% |
| Weighted F1-Score | 95-98% |
| Top-3 Accuracy | 99%+ |

### Generated Visualizations

1. **Training Curves** - Loss and accuracy over epochs
2. **Confusion Matrix** - Per-class prediction breakdown
3. **Class Distribution** - Dataset balance verification
4. **Prediction Confidence** - Model certainty analysis
5. **Sample Predictions** - Visual verification of predictions

---

## ⚙️ Configuration

Edit `config.json` to customize:

```json
{
  "dataset": {
    "name": "EuroSAT",
    "num_classes": 10,
    "image_size": [64, 64],
    "class_names": [...]
  },
  "model": {
    "input_shape": [64, 64, 3],
    "num_classes": 10,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "validation_split": 0.2,
    "test_split": 0.1
  },
  "training": {
    "use_data_augmentation": true,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-6
  },
  "paths": {
    "data_dir": "data/EuroSAT",
    "model_save_path": "models/satellite_cnn_model.h5",
    "plots_dir": "outputs/plots",
    "predictions_dir": "outputs/predictions"
  },
  "gui": {
    "window_title": "Satellite Image Classification System",
    "window_size": "900x700",
    "supported_formats": [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
  }
}
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. TensorFlow GPU not detected
```bash
# Install GPU version
pip install tensorflow[and-cuda]
# Or check CUDA installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### 2. Dataset download fails
```bash
# Manual download:
# 1. Go to https://madm.dfki.de/files/sentinel/EuroSAT.zip
# 2. Download and extract to data/EuroSAT/
# 3. Ensure folder structure: data/EuroSAT/ClassName/*.jpg
```

#### 3. Memory errors during training
```json
// In config.json, reduce batch_size:
"batch_size": 16
```

#### 4. Model not found error
```bash
# Train first:
python main.py train
# Or specify model path:
python main.py predict --image test.jpg --model models/satellite_cnn_model.h5
```

#### 5. GUI won't start
```bash
# Install tkinter (usually included with Python)
# On some systems:
# sudo apt-get install python3-tk  # Linux
# Or reinstall Python with "tcl/tk and IDLE" option checked
```

#### 6. Import errors
```bash
# Ensure you're in project root and venv is active
cd "C:\Users\omkaw\OneDrive\Desktop\health\SATELLITE IMAGE AND DETECTION"
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📚 For College Mini-Project

### Report Structure Suggestion
1. **Introduction** - Problem statement, objectives
2. **Literature Review** - CNN for satellite imagery, EuroSAT
3. **Methodology** - Dataset, preprocessing, CNN architecture
4. **Implementation** - Tools, libraries, code structure
5. **Results** - Accuracy, confusion matrix, training curves
6. **Discussion** - Performance analysis, limitations
7. **Conclusion** - Summary, future work

### Key Talking Points
- **Why CNN?** Spatial feature extraction, translation invariance
- **Why EuroSAT?** Standard benchmark, 10 diverse classes, Sentinel-2 data
- **Data Augmentation** - Critical for satellite imagery (rotation invariance)
- **BatchNorm + Dropout** - Regularization for better generalization
- **Global Average Pooling** - Reduces overfitting vs Flatten

### Demo Script
```bash
# Quick demo for presentation
python main.py gui
# Upload sample images from data/EuroSAT/*/ (any class folder)
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is for educational purposes (college mini-project).
EuroSAT dataset license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

---

## 🙏 Acknowledgments

- **EuroSAT Dataset**: Patrick Helber et al. - "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification"
- **TensorFlow/Keras Team** - Deep learning framework
- **Sentinel-2** - European Space Agency (ESA) for satellite imagery

---

## 📞 Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting)
2. Review TensorFlow/Keras documentation
3. Open an issue with error logs and system info

---

**Built with ❤️ for Satellite Image Analysis Education**