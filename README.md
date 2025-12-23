# Medical Ai Detecor 

---

## Abstract

The Medical Image AI Diagnostic System is a user-friendly web application that simplifies medical image analysis for healthcare professionals. It provides precise results for chest X-rays, brain MRIs, and skin lesions, showing clear confidence scores, image preparation details, and practical recommendations. Doctors can easily monitor analysis history, model performance, and patient records from a single intuitive dashboard, eliminating the need for manual initial image interpretation. Using advanced AI models like ResNet50, EfficientNet-B4, and DenseNet121, the system automatically processes uploaded images and generates detailed predictions with visual probability charts and downloadable reports. The tabbed interface offers Detection for real-time analysis, Training Pipeline for model insights, and Implementation for complete source code, creating the experience of an intelligent radiology assistant.Designed for busy clinical environments, it saves time on preliminary screenings while maintaining professional medical disclaimers that emphasize expert validation. The system transforms complex diagnostic workflows into seamless, automated processes, enabling healthcare providers to focus on patient care rather than image interpretation.

---

## Results

### 1. Detection Interface
![Detection Interface](./screenshots/Detection%20Interface.png)

### 2. Image Upload & Analysis
![Image Upload](./screenshots/Image%20Upload.png)

### AI Analysis Results
![Analysis Results](./screenshots/Analysis%20Results.png)


# Tech Stack

## Frontend
- React
- Lucide React
- Tailwind CSS
- JavaScript

## Browser APIs
- FileReader API
- Blob API
- URL API

## Backend 
- Python
- TensorFlow
- Keras
- NumPy
- scikit-learn
- Pillow
- OpenCV

## Deep Learning Models
- ResNet50 (X-Ray Classification)
- EfficientNet-B4 (MRI Analysis)
- DenseNet121 (Skin Lesion Detection)

## API & Deployment
- Flask / FastAPI
- Docker
- ONNX Runtime
- CUDA

## Data Processing
- ImageDataGenerator
- DICOM Support
- Data Augmentation

## Training Components
- Adam Optimizer
- Cross-Entropy Loss
- Early Stopping
- Model Checkpointing

---

## Users

1. Radiologists 
2. Dermatologists
3. Emergency Doctors

---
### 1. Clone the repository
```bash
git clone https://github.com/bakshitha07/medical-image-ai-diagnostic-system.git
cd medical-image-ai-diagnostic-system
```

### 2. Install frontend dependencies
```bash
npm install

### 3. Install and configure Tailwind CSS
```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### 4. Start the application
```bash
npm start
```


</div>
```
