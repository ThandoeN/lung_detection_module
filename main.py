import cv2
import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_processor import LungImageProcessor
from utils import display_images, load_image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    directories = ['data/sample_images', 'data/results', 'logs']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("Directories set up successfully")

def process_single_image(image_path):
    """Process a single chest X-ray image with enhanced error handling"""
    logger.info(f"Starting processing for: {image_path}")
    
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None, 0
    
    try:
        # Initialize processor
        processor = LungImageProcessor()
        
        # Load image
        logger.debug("Loading image...")
        original = processor.load_image(image_path)
        logger.info(f"Image loaded: {original.shape}")
        
        # Preprocess
        logger.debug("Preprocessing image...")
        processed = processor.preprocess_image(original)
        
        # Detect nodules using multiple methods
        logger.debug("Detecting nodules...")
        keypoints = processor.detect_nodules(processed)
        contours = processor.analyze_regions(processed)
        
        # Draw results
        logger.debug("Drawing detection results...")
        result_image = processor.draw_detections(original, keypoints, contours)
        
        # Save result
        output_path = f"data/results/result_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, result_image)
        
        # Display results
        display_images(original, result_image, 
                      f"Original: {os.path.basename(image_path)}", 
                      f"Detections: {len(keypoints) + len(contours)} found")
        
        total_detections = len(keypoints) + len(contections)
        logger.info(f"Processing completed. Total detections: {total_detections}")
        
        return result_image, total_detections
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}", exc_info=True)
        return None, 0

def main():
    """Main function with menu options"""
    setup_directories()
    
    print("=" * 50)
    print("   Lung Nodule Detection - VS Code Edition")
    print("=" * 50)
    print("\nOptions:")
    print("1. Test with synthetic image")
    print("2. Process a single X-ray image")
    print("3. Process multiple images in a folder")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            from test_image_creator import test_with_synthetic_image
            test_with_synthetic_image()
            
        elif choice == "2":
            image_path = input("Enter the path to your X-ray image: ").strip()
            process_single_image(image_path)
            
        elif choice == "3":
            folder_path = input("Enter the path to your images folder: ").strip()
            # You can implement batch processing here
            print("Batch processing would go here...")
            
        elif choice == "4":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main() 
    