import numpy as np
import cv2
import os
from image_processor import LungImageProcessor
from utils import display_images

def create_test_xray():
    """Create a synthetic chest X-ray for testing"""
    # Create a dark background (like X-ray)
    height, width = 512, 512
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Add "lungs" - darker oval regions
    cv2.ellipse(image, (width//3, height//2), (80, 120), 0, 0, 360, 100, -1)
    cv2.ellipse(image, (2*width//3, height//2), (80, 120), 0, 0, 360, 100, -1)
    
    # Add "ribs" - brighter curved lines
    for i in range(5):
        y = height//3 + i*30
        cv2.ellipse(image, (width//2, y), (200, 20), 0, 0, 360, 200, 2)
    
    # Add some "nodules" - small bright circles
    cv2.circle(image, (width//3, height//2), 8, 220, -1)  # Potential nodule
    cv2.circle(image, (2*width//3, height//2 + 40), 6, 210, -1)  # Another nodule
    cv2.circle(image, (width//2, height//2), 4, 200, -1)  # Small nodule
    
    # Add some noise to make it realistic
    noise = np.random.normal(0, 10, (height, width)).astype(np.uint8)
    image = cv2.add(image, noise)
    
    return image

def clear_nodules(image):
    """Remove the default nodules by recreating the base image"""
    height, width = image.shape
    new_image = np.zeros((height, width), dtype=np.uint8)
    
    # Recreate base structure without nodules
    cv2.ellipse(new_image, (width//3, height//2), (80, 120), 0, 0, 360, 100, -1)
    cv2.ellipse(new_image, (2*width//3, height//2), (80, 120), 0, 0, 360, 100, -1)
    
    for i in range(5):
        y = height//3 + i*30
        cv2.ellipse(new_image, (width//2, y), (200, 20), 0, 0, 360, 200, 2)
    
    noise = np.random.normal(0, 10, (height, width)).astype(np.uint8)
    new_image = cv2.add(new_image, noise)
    
    return new_image

def create_multiple_test_images():
    """Create multiple test images with different characteristics"""
    height, width = 512, 512  # Define width and height here
    
    variations = [
        ("large_nodules", [(width//3, height//2, 12), (2*width//3, height//2, 10)]),
        ("small_nodules", [(width//4, height//3, 4), (3*width//4, 2*height//3, 5)]),
        ("multiple_nodules", [(width//3, height//3, 6), (2*width//3, height//3, 7), 
                             (width//2, 2*height//3, 5), (width//4, 3*height//4, 6)])
    ]
    
    for name, nodules in variations:
        image = create_test_xray()
        
        # Clear default nodules and add specific ones
        image = clear_nodules(image)
        for x, y, size in nodules:
            cv2.circle(image, (x, y), size, 220, -1)
        
        cv2.imwrite(f'data/sample_images/test_{name}.png', image)
        print(f"Created: data/sample_images/test_{name}.png")

def test_with_synthetic_image():
    """Test our pipeline with a synthetic image"""
    print("Creating synthetic chest X-ray for testing...")
    
    # Create test image
    test_image = create_test_xray()
    
    # Create directory if it doesn't exist
    os.makedirs('data/sample_images', exist_ok=True)
    
    # Save it temporarily
    cv2.imwrite('data/sample_images/test_xray.png', test_image)
    
    # Process it
    processor = LungImageProcessor()
    
    # Use the image directly (since we have it in memory)
    original = test_image
    processed = processor.preprocess_image(original)
    
    # Detect nodules
    keypoints = processor.detect_nodules(processed)
    contours = processor.analyze_regions(processed)
    
    # Draw results
    result_image = processor.draw_detections(original, keypoints, contours)
    
    # Display
    display_images(original, result_image, "Synthetic X-ray", "Detected Nodules")
    
    print(f"Detection results:")
    print(f"- Blob detection found {len(keypoints)} potential nodules")
    print(f"- Contour analysis found {len(contours)} potential nodules")
    
    # Ask if user wants to keep the test image
    keep = input("\nKeep the test image? (y/n): ").strip().lower()
    if keep != 'y':
        if os.path.exists('data/sample_images/test_xray.png'):
            os.remove('data/sample_images/test_xray.png')
            print("Test image deleted.")
    else:
        print("Test image saved at: data/sample_images/test_xray.png")

if __name__ == "__main__":
    print("Test Image Creator for Lung Nodule Detection")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Create single test image and run detection")
        print("2. Create multiple test image variations")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            test_with_synthetic_image()
        elif choice == "2":
            create_multiple_test_images()
            print("\nMultiple test images created successfully!")
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")