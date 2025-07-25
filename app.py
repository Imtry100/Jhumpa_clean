from flask import Flask, render_template, request, jsonify, send_file
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import joblib
import torch.nn as nn
import os
import base64
from io import BytesIO
import time
import streamlit as st
from werkzeug.utils import secure_filename

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolo_flask_app', 'yolov7'))

# YOLOv7 imports (assuming you have the YOLOv7 structure)
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xyxy2xywh, check_img_size
from utils.torch_utils import select_device

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SkinAnalysisModels:
    def __init__(self):
        self.device = device
        self.load_models()
        
    def load_models(self):
        """Load all three models"""
        try:
            # Load Skin Disease Model - UPDATED FOR NEW MODEL
            # Load class_to_idx mapping instead of label encoder
            self.class_to_idx = joblib.load(os.path.join('yolo_flask_app', 'class_to_idx.pkl'))
            # Create reverse mapping (index to class name)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            
            # Load model with 11 classes (updated from 10)
            self.skin_disease_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
            self.skin_disease_model.classifier[1] = torch.nn.Linear(self.skin_disease_model.classifier[1].in_features, 11)  # Changed from 10 to 11
            self.skin_disease_model.load_state_dict(torch.load(os.path.join('yolo_flask_app', 'best_model_efficientnetv2.pth'), map_location=self.device))
            self.skin_disease_model = self.skin_disease_model.to(self.device)
            self.skin_disease_model.eval()
            
            # Load Skin Type Model (unchanged)
            self.skin_type_le = joblib.load(os.path.join('yolo_flask_app', 'skin_type_label_encoder.pkl'))
            self.skin_type_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
            self.skin_type_model.classifier[1] = nn.Linear(self.skin_type_model.classifier[1].in_features, len(self.skin_type_le.classes_))
            self.skin_type_model.load_state_dict(torch.load(os.path.join('yolo_flask_app', 'best_model_skin_type_efficientnetv2.pth'), map_location=self.device))
            self.skin_type_model = self.skin_type_model.to(self.device)
            self.skin_type_model.eval()
            
            # Load Pimple Detection Model (YOLOv7) (unchanged)
            self.pimple_weights = os.path.join('yolo_flask_app','yolov7','runs','train','pipmle_50_fresh20','weights','last.pt')
            self.pimple_model = attempt_load(self.pimple_weights, map_location=self.device)
            self.pimple_model.eval()
            
            # Image transforms (unchanged)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("All models loaded successfully!")
            print(f"Skin disease classes loaded: {list(self.class_to_idx.keys())}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def flatten_image(self, image):
        """Flatten lighting and contrast for better model performance"""
        if isinstance(image, Image.Image):
            image = np.array(image)

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(final)
    
    def refine_image(self, image):
        """Refine image by removing hair and artifacts"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        hair_removed = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
        
        kernel_morph = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(hair_removed, cv2.MORPH_CLOSE, kernel_morph)
        
        return Image.fromarray(morph)
    
    def predict_skin_disease(self, image):
        """Predict skin disease - UPDATED FOR NEW MODEL STRUCTURE"""
        try:
            flattened = self.flatten_image(image)
            refined_image = self.refine_image(flattened)
            image_tensor = self.transform(refined_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.skin_disease_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(outputs, 1)
                
                # Use idx_to_class mapping instead of label encoder
                try:
                    predicted_label = self.idx_to_class[predicted_idx.item()]
                except KeyError:
                    predicted_label = "Unknown"
                    
                confidence_score = probabilities[predicted_idx].item()
                
            return predicted_label, confidence_score
        except Exception as e:
            print(f"Error in skin disease prediction: {e}")
            return "Unknown", 0.0
    
    def predict_skin_type(self, image):
        """Predict skin type (unchanged)"""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            flattened = self.flatten_image(image)
            image_tensor = self.transform(flattened).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.skin_type_model(image_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted = torch.max(output, 1)
                try:
                    predicted_class = self.skin_type_le.inverse_transform(predicted.cpu().numpy())[0]
                except:
                    predicted_class = "Unknown"

                confidence_score = probabilities[predicted].item()
                
            return predicted_class, confidence_score
        except Exception as e:
            print(f"Error in skin type prediction: {e}")
            return "Unknown", 0.0
    
    def detect_pimples(self, image, img_size=416, conf_thres=0.25, iou_thres=0.45):
        """Detect pimples using YOLOv7 (unchanged)"""
        try:
            # Convert PIL to opencv format
            if isinstance(image, Image.Image):
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Prepare image for YOLO
            img_size = check_img_size(img_size, s=self.pimple_model.stride.max())
            img = cv2.resize(image, (img_size, img_size))
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img, dtype=np.float32)
            
            # Convert to tensor
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                pred = self.pimple_model(img)[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres)
            
            # Process detections
            detections = []
            annotated_image = image.copy()
            
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes from img_size to original image size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                    
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = map(int, xyxy)
                        confidence = conf.item()
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class': 'pimple'
                        })
                        
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_image, f'Pimple {confidence:.2f}', 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            return detections, Image.fromarray(annotated_image)
            
        except Exception as e:
            print(f"Error in pimple detection: {e}")
            if isinstance(image, np.ndarray):
                try:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except:
                    pass
                return [], Image.fromarray(image)
            elif isinstance(image, Image.Image):
                return [], image
            else:
                raise ValueError("Unsupported image format for annotated output.")


# Initialize models
skin_models = SkinAnalysisModels()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_skin():
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        if file.content_length and file.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'Image size too large'}), 413
        
        if file.content_length and file.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'Image size too large'}), 413

        
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            image = Image.open(filepath).convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Invalid image format: {str(e)}'}), 400
        
        skin_disease, disease_confidence = skin_models.predict_skin_disease(image)
        skin_disease_result = {
            'prediction': skin_disease,
            'confidence': f"{disease_confidence:.2%}"
            }
        skin_type, type_confidence = skin_models.predict_skin_type(image)
        pimple_detections, annotated_image = skin_models.detect_pimples(image)
        
        annotated_filename = f"annotated_{filename}"
        annotated_filepath = os.path.join(app.config['RESULTS_FOLDER'], annotated_filename)
        if isinstance(annotated_image, np.ndarray):
            annotated_image = Image.fromarray(annotated_image)
        annotated_image.save(annotated_filepath)
        
        def image_to_base64(img):
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        
        original_b64 = image_to_base64(image)
        annotated_b64 = image_to_base64(annotated_image)
        
        results = {
             'skin_disease': skin_disease_result,
            'skin_type': {
                'prediction': skin_type,
                'confidence': f"{type_confidence:.2%}"
            },
            'pimple_detection': {
                'count': len(pimple_detections),
                'detections': pimple_detections
            },
            'images': {
                'original': original_b64,
                'annotated': annotated_b64
            }
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'device': str(device)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
