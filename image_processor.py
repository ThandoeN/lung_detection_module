import cv2
import numpy as np

class LungImageProcessor:
    def _init_(self):
        self.original_image = None
        self.processed_image = None
        
    def load_image(self, image_path):
        """Load and store the original image"""
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return self.original_image
    
    def preprocess_image(self, image):
        """Enhance image quality and contrast"""
        # Normalize image
        normalized = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply((normalized * 255).astype(np.uint8))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        self.processed_image = blurred
        return blurred
    
    def segment_lungs(self, image):
        """Simple lung region segmentation using thresholding"""
        # Apply Otsu's threshold
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up the mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def detect_nodules(self, image, min_radius=3, max_radius=20):
        """Detect potential nodules using blob detection"""
        # Use SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by area
        params.filterByArea = True
        params.minArea = np.pi * min_radius ** 2
        params.maxArea = np.pi * max_radius ** 2
        
        # Filter by circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.8
        
        # Create detector and detect blobs
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)
        
        return keypoints
    
    def analyze_regions(self, image):
        """Alternative method using region analysis"""
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_nodules = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area (potential nodule size)
            if 50 < area < 1000:  # Adjust these values based on your image resolution
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.5:  # Nodules are often round
                        potential_nodules.append(contour)
        
        return potential_nodules
    
    def draw_detections(self, image, keypoints, contours=None):
        """Draw detected nodules on the image"""
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw blob keypoints
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            cv2.circle(result, (x, y), radius, (0, 0, 255), 2)  # Red circles
        
        # Draw contour-based detections
        if contours:
            for contour in contours:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(result, center, radius, (255, 0, 0), 2)  # Blue circles
        
        return result