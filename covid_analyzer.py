import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from image_processor import LungImageProcessor
from utils import display_images, load_image
from covid_loader import CovidDatasetLoader

class CovidImageAnalyzer:
    def __init__(self):
        self.processor = LungImageProcessor()
        self.loader = CovidDatasetLoader()
        self.results = []
        
    def analyze_single_image(self, image_path, category="Unknown"):
        """Analyze a single COVID dataset image"""
        print(f"\n🔍 Analyzing {category} image: {os.path.basename(image_path)}")
        
        try:
            # Load the image
            original = load_image(image_path)
            print(f"   Image size: {original.shape}")
            
            # Enhanced preprocessing for COVID images
            processed = self.enhanced_covid_preprocessing(original)
            
            # Detect nodules
            keypoints = self.processor.detect_nodules(processed)
            contours = self.processor.analyze_regions(processed)
            
            # Draw results
            result_image = self.processor.draw_detections(original, keypoints, contours)
            
            # Display results
            display_images(original, result_image, 
                          f"Original: {category}", 
                          f"Detections: {len(keypoints)} blobs, {len(contours)} contours")
            
            # Save analysis results
            analysis_result = {
                'category': category,
                'filename': os.path.basename(image_path),
                'image_size': original.shape,
                'blob_detections': len(keypoints),
                'contour_detections': len(contours),
                'total_findings': len(keypoints) + len(contours),
                'image_path': image_path
            }
            
            self.results.append(analysis_result)
            self.print_analysis_report(analysis_result)
            
            return result_image, len(keypoints), len(contours)
            
        except Exception as e:
            print(f"❌ Error analyzing {image_path}: {e}")
            return None, 0, 0
    
    def enhanced_covid_preprocessing(self, image):
        """Enhanced preprocessing optimized for COVID dataset images"""
        # Step 1: Resize if too large (for performance)
        h, w = image.shape
        if w > 1024:
            scale = 1024 / w
            new_width = 1024
            new_height = int(h * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Step 2: Normalize
        normalized = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        # Step 3: CLAHE optimized for chest X-rays
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply((normalized * 255).astype(np.uint8))
        
        # Step 4: Noise reduction
        denoised = cv2.medianBlur(enhanced, 3)
        
        return denoised
    
    def analyze_category(self, category, max_images=3):
        """Analyze multiple images from a specific category"""
        print(f"\n🎯 ANALYZING {category.upper()} IMAGES")
        print("=" * 50)
        
        image_paths = self.loader.get_all_category_images(category, max_images)
        
        if not image_paths:
            print(f"❌ No images found in {category} category")
            return 0, 0
        
        total_blobs = 0
        total_contours = 0
        
        for i, image_path in enumerate(image_paths):
            result, blobs, contours = self.analyze_single_image(image_path, category)
            total_blobs += blobs
            total_contours += contours
        
        print(f"\n📊 {category.upper()} CATEGORY SUMMARY:")
        print(f"   Images analyzed: {len(image_paths)}")
        print(f"   Total blob detections: {total_blobs}")
        print(f"   Total contour detections: {total_contours}")
        print(f"   Average findings per image: {(total_blobs + total_contours) / len(image_paths):.1f}")
        
        return total_blobs, total_contours
    
    def compare_all_categories(self, images_per_category=2):
        """Compare findings across all COVID dataset categories"""
        print("🦠 COMPARING COVID DATASET CATEGORIES")
        print("=" * 50)
        
        category_results = {}
        
        for category in ['COVID', 'Normal', 'Lung_Opacity', 'Viral_Pneumonia']:
            print(f"\n{'='*30}")
            blobs, contours = self.analyze_category(category, images_per_category)
            category_results[category] = {
                'blobs': blobs,
                'contours': contours,
                'total_findings': blobs + contours
            }
        
        # Create comparison report
        self.create_comparison_report(category_results, images_per_category)
    
    def create_comparison_report(self, category_results, images_per_category):
        """Create a visual comparison report"""
        print("\n📈 COMPARISON REPORT")
        print("=" * 30)
        
        categories = list(category_results.keys())
        total_findings = [category_results[cat]['total_findings'] for cat in categories]
        blobs = [category_results[cat]['blobs'] for cat in categories]
        contours = [category_results[cat]['contours'] for cat in categories]
        
        # Print table
        print(f"{'Category':<15} {'Images':<8} {'Blobs':<8} {'Contours':<10} {'Total':<8} {'Avg/Image':<10}")
        print("-" * 65)
        for category in categories:
            result = category_results[category]
            avg_per_image = result['total_findings'] / images_per_category
            print(f"{category:<15} {images_per_category:<8} {result['blobs']:<8} {result['contours']:<10} {result['total_findings']:<8} {avg_per_image:<10.1f}")
        
        # Create bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Total findings plot
        bars = ax1.bar(categories, total_findings, color=['red', 'green', 'blue'])
        ax1.set_title('Total Findings by Category')
        ax1.set_ylabel('Number of Detections')
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Detection type breakdown
        x = np.arange(len(categories))
        width = 0.35
        ax2.bar(x - width/2, blobs, width, label='Blob Detections', color='orange')
        ax2.bar(x + width/2, contours, width, label='Contour Detections', color='purple')
        ax2.set_title('Detection Type Breakdown')
        ax2.set_ylabel('Number of Detections')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Save results to CSV
        self.save_results_to_csv()
    
    def print_analysis_report(self, result):
        """Print individual image analysis report"""
        print(f"📊 ANALYSIS REPORT:")
        print(f"   Category: {result['category']}")
        print(f"   Image: {result['filename']}")
        print(f"   Size: {result['image_size']}")
        print(f"   Blob detections: {result['blob_detections']}")
        print(f"   Contour detections: {result['contour_detections']}")
        print(f"   Total findings: {result['total_findings']}")
    
    def save_results_to_csv(self):
        """Save all results to CSV file"""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        csv_path = "data/results/covid_analysis_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Results saved to: {csv_path}")

def main():
    """Main function for COVID dataset analysis"""
    analyzer = CovidImageAnalyzer()
    
    print("🦠 COVID-19 CHEST X-RAY ANALYZER")
    print("=" * 50)
    
    # First, show dataset info
    analyzer.loader.get_dataset_info()
    
    while True:
        print("\nOptions:")
        print("1. Analyze COVID-positive images")
        print("2. Analyze Normal (healthy) images") 
        print("3. Analyze Lung Opacity images")
        print("4. Compare all categories")
        print("5. Preview dataset images")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            try:
                count = int(input("How many COVID images to analyze? (default 2): ") or "2")
                analyzer.analyze_category('COVID', count)
            except ValueError:
                print("❌ Please enter a valid number!")
                
        elif choice == "2":
            try:
                count = int(input("How many Normal images to analyze? (default 2): ") or "2")
                analyzer.analyze_category('Normal', count)
            except ValueError:
                print("❌ Please enter a valid number!")
                
        elif choice == "3":
            try:
                count = int(input("How many Lung Opacity images? (default 2): ") or "2")
                analyzer.analyze_category('Lung_Opacity', count)
            except ValueError:
                print("❌ Please enter a valid number!")
                
        elif choice == "4":
            try:
                count = int(input("Images per category? (default 2): ") or "2")
                analyzer.compare_all_categories(count)
            except ValueError:
                print("❌ Please enter a valid number!")
                
        elif choice == "5":
            analyzer.loader.preview_all_categories()
            
        elif choice == "6":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()