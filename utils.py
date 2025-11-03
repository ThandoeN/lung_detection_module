import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_image(image_path):
    """Load an image in grayscale"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image

def display_images(original, processed, title1="Original", title2="Processed"):
    """Display original and processed images side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title(title1)
    ax1.axis('off')
    
    ax2.imshow(processed, cmap='gray')
    ax2.set_title(title2)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def normalize_image(image):
    """Normalize pixel values to 0-1 range"""
    return cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

def save_image(image, output_path):
    """Save an image to file"""
    cv2.imwrite(output_path, image)
    print(f"Image saved to: {output_path}")

def resize_image(image, width=None, height=None):
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized