import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import requests
import zipfile
from pathlib import Path


class EuroSATDataLoader:
    """Handles downloading, loading, and preprocessing of EuroSAT dataset."""
    
    def __init__(self, config_path="config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.dataset_config = self.config['dataset']
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.paths = self.config['paths']
        
        self.data_dir = Path(self.paths['data_dir'])
        self.image_size = tuple(self.dataset_config['image_size'])
        self.num_classes = self.dataset_config['num_classes']
        self.class_names = self.dataset_config['class_names']
        self.batch_size = self.model_config['batch_size']
        self.validation_split = self.model_config['validation_split']
        self.test_split = self.model_config['test_split']
        self.use_augmentation = self.training_config['use_data_augmentation']
    
    def download_dataset(self):
        """Download and extract EuroSAT dataset if not present."""
        zip_path = self.data_dir.parent / "EuroSAT.zip"
        
        if self.data_dir.exists() and any(self.data_dir.iterdir()):
            print(f"Dataset already exists at {self.data_dir}")
            return True
        
        print("Downloading EuroSAT dataset (~2 GB)...")
        print("This may take several minutes depending on your connection.")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            url = self.dataset_config['url']
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir.parent)
            
            zip_path.unlink()
            print(f"Dataset extracted to {self.data_dir}")
            return True
            
        except requests.exceptions.Timeout:
            print("Download timed out. The EuroSAT server may be slow or unreachable.")
            print("Please manually download from: https://madm.dfki.de/files/sentinel/EuroSAT.zip")
            print(f"Extract to: {self.data_dir.parent}")
            return False
        except requests.exceptions.ConnectionError:
            print("Connection error. Please check your internet connection.")
            print("Manual download: https://madm.dfki.de/files/sentinel/EuroSAT.zip")
            print(f"Extract to: {self.data_dir.parent}")
            return False
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Manual download: https://madm.dfki.de/files/sentinel/EuroSAT.zip")
            print(f"Extract to: {self.data_dir.parent}")
            return False
    
    def load_images_from_directory(self):
        """Load images and labels from directory structure."""
        images = []
        labels = []
        
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if len(class_dirs) != self.num_classes:
            print(f"Warning: Expected {self.num_classes} classes, found {len(class_dirs)}")
        
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in class_to_idx:
                print(f"Skipping unknown class: {class_name}")
                continue
            
            class_idx = class_to_idx[class_name]
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.tif"))
            
            for img_path in tqdm(image_files, desc=f"Loading {class_name}"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.image_size)
                    img = img.astype(np.float32) / 255.0
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Create data generators with augmentation for training."""
        
        if self.use_augmentation:
            train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.2,
                shear_range=0.1,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator()
        
        val_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        test_generator = test_datagen.flow(
            X_test, y_test,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    def create_synthetic_data(self, num_samples_per_class=100):
        """Create synthetic data for testing when real dataset is unavailable."""
        print("Creating synthetic test data...")
        np.random.seed(42)
        
        images = []
        labels = []
        
        for class_idx in range(self.num_classes):
            for _ in range(num_samples_per_class):
                # Create synthetic image with some class-specific patterns
                img = np.random.rand(*self.image_size, 3).astype(np.float32)
                
                # Add class-specific color bias for minimal distinguishability
                if class_idx == 0:  # AnnualCrop - greenish
                    img[:, :, 1] += 0.3
                elif class_idx == 1:  # Forest - darker green
                    img[:, :, 1] += 0.4
                    img *= 0.7
                elif class_idx == 2:  # HerbaceousVegetation - yellowish green
                    img[:, :, 0] += 0.2
                    img[:, :, 1] += 0.3
                elif class_idx == 3:  # Highway - gray
                    img[:, :] = 0.5
                elif class_idx == 4:  # Industrial - bluish gray
                    img[:, :, 2] += 0.2
                    img *= 0.6
                elif class_idx == 5:  # Pasture - light green
                    img[:, :, 1] += 0.2
                    img *= 0.8
                elif class_idx == 6:  # PermanentCrop - row patterns
                    img[::4, :] = 0.8
                    img[:, :, 1] += 0.2
                elif class_idx == 7:  # Residential - reddish
                    img[:, :, 0] += 0.3
                    img *= 0.7
                elif class_idx == 8:  # River - blue
                    img[:, :, 2] += 0.5
                    img *= 0.5
                elif class_idx == 9:  # SeaLake - deep blue
                    img[:, :, 2] += 0.6
                    img *= 0.4
                
                img = np.clip(img, 0, 1)
                images.append(img)
                labels.append(class_idx)
        
        return np.array(images), np.array(labels)
    
    def prepare_data(self, use_synthetic=False):
        """Main method to prepare all data splits."""
        print("Preparing EuroSAT dataset...")
        
        if use_synthetic or not self.download_dataset():
            if not use_synthetic:
                print("\nUsing synthetic data for testing...")
            X, y = self.create_synthetic_data(num_samples_per_class=200)
        else:
            print("Loading images...")
            X, y = self.load_images_from_directory()
        
        if len(X) == 0:
            raise ValueError("No images loaded. Check dataset path and structure.")
        
        print(f"Loaded {len(X)} images")
        
        y_categorical = to_categorical(y, self.num_classes)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical, 
            test_size=self.test_split, 
            random_state=42, 
            stratify=y
        )
        
        val_size = self.validation_split / (1 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=42,
            stratify=np.argmax(y_temp, axis=1)
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        train_gen, val_gen, test_gen = self.create_data_generators(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        return {
            'train': (X_train, y_train, train_gen),
            'val': (X_val, y_val, val_gen),
            'test': (X_test, y_test, test_gen),
            'class_names': self.class_names
        }
    
    def load_single_image(self, image_path):
        """Load and preprocess a single image for prediction."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_img = img.copy()
            img = cv2.resize(img, self.image_size)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            return img, original_img
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")


def get_data_loader(config_path="config.json"):
    """Factory function to get data loader instance."""
    return EuroSATDataLoader(config_path)