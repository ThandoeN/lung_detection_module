import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import display_images

class CovidDatasetLoader:
    def __init__(self):
        self.dataset_path = "data/covid_dataset"
        # Note the actual folder names in your dataset
        self.categories = {
            'COVID': 'COVID',
            'Normal': 'normal',  # lowercase in your folder
            'Lung_Opacity': 'lung_opacity',  # lowercase with underscore
            'Viral_Pneumonia': 'Viral Pneumonia'  # space in folder name
        }
        
    def get_dataset_info(self):
        """Get information about the COVID dataset"""
        print("🦠 COVID-19 DATASET INFORMATION")
        print("=" * 50)
        
        if not os.path.exists(self.dataset_path):
            print("❌ COVID dataset folder not found!")
            print(f"   Expected path: {self.dataset_path}")
            return None
            
        category_info = {}
        total_images = 0
        
        for display_name, folder_name in self.categories.items():
            # Images are in the "images" subfolder of each category
            images_path = os.path.join(self.dataset_path, folder_name, "images")
            
            if os.path.exists(images_path):
                # Count image files
                images = [f for f in os.listdir(images_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                image_count = len(images)
                category_info[display_name] = {
                    'count': image_count,
                    'path': images_path,
                    'images': images,
                    'folder_name': folder_name
                }
                total_images += image_count
                print(f"📁 {display_name}: {image_count} images")
            else:
                print(f"❌ Images folder not found: {images_path}")
        
        print(f"\n📊 TOTAL IMAGES: {total_images}")
        return category_info
    
    def get_all_category_images(self, category, max_images=None):
        """Get all image paths from a category"""
        category_info = self.get_dataset_info()
        if not category_info or category not in category_info:
            return []
            
        images_path = category_info[category]['path']
        images = category_info[category]['images']
        
        if max_images:
            images = images[:max_images]
            
        return [os.path.join(images_path, img) for img in images]
    
    def preview_category_images(self, category, num_images=3):
        """Preview images from a specific category"""
        print(f"\n🖼️  Previewing {category} images:")
        
        image_paths = self.get_all_category_images(category, num_images)
        
        if not image_paths:
            print(f"❌ No images found in {category} category")
            return
            
        # Create subplots
        if len(image_paths) == 1:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            axes = [axes]
        else:
            fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
            
        for i, image_path in enumerate(image_paths):
            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    axes[i].imshow(image, cmap='gray')
                    axes[i].set_title(f"{category}\n{os.path.basename(image_path)}")
                    axes[i].axis('off')
                    
                    # Print image info
                    h, w = image.shape
                    print(f"   {i+1}. {os.path.basename(image_path)}: {w}x{h} pixels")
                else:
                    print(f"   ❌ Could not load: {os.path.basename(image_path)}")
                    # Show placeholder
                    axes[i].text(0.5, 0.5, "Load Error", ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f"{category}\nLoad Error")
                    axes[i].axis('off')
            except Exception as e:
                print(f"   ❌ Error loading image: {e}")
                axes[i].text(0.5, 0.5, "Error", ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{category}\nError")
                axes[i].axis('off')
                
        plt.tight_layout()
        plt.show()
    
    def preview_all_categories(self):
        """Preview all categories in the dataset"""
        category_info = self.get_dataset_info()
        if not category_info:
            return
            
        for category in category_info.keys():
            self.preview_category_images(category, 2)
    
    def get_image_stats(self, image_path):
        """Get statistics for an image"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
            
        return {
            'path': image_path,
            'shape': image.shape,
            'mean_intensity': np.mean(image),
            'std_intensity': np.std(image),
            'min_intensity': np.min(image),
            'max_intensity': np.max(image)
        }

def test_covid_loader():
    """Test the COVID dataset loader"""
    loader = CovidDatasetLoader()
    loader.preview_all_categories()

if __name__ == "__main__":
    test_covid_loader()