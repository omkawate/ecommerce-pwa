import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from train import main as train_main
from predict import main as predict_main
from gui.app import main as gui_main


def print_banner():
    print("""
================================================================================
                                                                             
   SATELLITE IMAGE DETECTION AND CLASSIFICATION SYSTEM                       
                                                                             
   CNN-based Land Use / Land Cover Classification using EuroSAT Dataset      
                                                                             
================================================================================
""")


def print_usage():
    print("""
USAGE:
    python main.py <command> [options]

COMMANDS:
    train       Train the CNN model on EuroSAT dataset
    predict     Run prediction on image(s) using trained model
    gui         Launch graphical user interface
    help        Show this help message

EXAMPLES:
    python main.py train
    python main.py predict --image data/test_image.jpg
    python main.py predict --batch data/img1.jpg data/img2.jpg
    python main.py gui

TRAINING OPTIONS:
    --config CONFIG       Path to config file (default: config.json)
    --epochs EPOCHS       Number of training epochs
    --batch-size SIZE     Batch size for training
    --lr RATE             Learning rate

PREDICTION OPTIONS:
    --image PATH          Single image to classify
    --batch PATHS...      Multiple images to classify
    --model PATH          Path to trained model (default: models/satellite_cnn_model.h5)
    --config CONFIG       Path to config file (default: config.json)

For more information on a specific command:
    python main.py <command> --help
""")


def main():
    print_banner()
    
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'train':
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        train_main()
    
    elif command == 'predict':
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        predict_main()
    
    elif command == 'gui':
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        gui_main()
    
    elif command in ['help', '-h', '--help']:
        print_usage()
    
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()