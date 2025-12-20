# 🏥 Medical Image AI Diagnostic System

An end-to-end deep learning web application for automated disease classification from medical images, integrating transfer-learning-based CNN models with a modern React frontend.

---

## 📌 Overview

This project demonstrates the design and implementation of a full-stack AI system capable of analyzing multiple medical imaging modalities, including chest X-rays, brain MRIs, and skin lesion images, to predict disease categories with confidence scores.

The system is built for educational and research purposes, focusing on model development, evaluation, and frontend–ML integration.

---

## ✨ Key Features

### 🩻 Multi-Modal Medical Image Classification

**Chest X-Ray Analysis**
- Pneumonia  
- COVID-19  
- Tuberculosis  
- Lung Cancer  
- Normal  

**Brain MRI Analysis**
- Glioma  
- Meningioma  
- Pituitary Tumor  
- Alzheimer’s Disease  
- Normal  

**Skin Lesion Classification**
- Melanoma  
- Basal Cell Carcinoma  
- Squamous Cell Carcinoma  
- Actinic Keratosis  
- Benign Lesions  

---

### 🤖 AI & Machine Learning Capabilities
- Transfer learning using pre-trained CNN architectures
- Real-time image preprocessing and inference
- Multi-class classification with confidence scoring
- Modular pipeline supporting multiple imaging modalities
- Downloadable diagnostic summary reports
- User-friendly web interface with drag-and-drop image upload

---

## 🛠️ Tech Stack

### Frontend
- React.js (18+)
- Tailwind CSS
- Lucide React

### Backend / Machine Learning
- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Scikit-learn

---

## 🧠 Model Architectures

| Imaging Type | Architecture | Classes | Validation Accuracy* |
|-------------|--------------|---------|----------------------|
| Chest X-Ray | ResNet50 | 5 | ~94% |
| Brain MRI | EfficientNet-B4 | 5 | ~91% |
| Skin Lesion | DenseNet121 | 5 | ~89% |

\*Accuracy measured on validation splits of publicly available academic datasets.

---

## 🏗️ Transfer Learning Pipeline

Pre-trained CNN (ImageNet)
↓
Global Average Pooling
↓
Dense Layer (512, ReLU)
↓
Dropout (0.5)
↓
Softmax Output Layer


---

## ⚙️ Training Configuration

- Optimizer: Adam (learning rate = 0.0001)
- Loss Function: Categorical Cross-Entropy
- Batch Size: 32
- Epochs: 50 (Early Stopping enabled)
- Data Augmentation: Rotation, flipping, zoom, brightness adjustment

---

## 🚀 Installation & Setup

### Prerequisites
- Node.js 16+
- npm
- Python 3.8+

---

### 1. Clone the repository
```bash
git clone https://github.com/bakshitha07/medical-image-ai-diagnostic-system.git
cd medical-image-ai-diagnostic-system 

2.Install frontend dependencies
npm install

3.Install and configure Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

4.Start the application
npm start


Open http://localhost:3000
 in your browser.

📁 Project Structure
medical-image-ai-diagnostic-system/
├── src/
│   ├── components/
│   │   └── MedicalImageDetector.jsx
│   ├── App.js
│   ├── index.js
│   └── index.css
├── public/
├── tailwind.config.js
├── postcss.config.js
├── package.json
└── README.md

🧪 Usage

1.Select imaging modality (X-Ray, MRI, or Skin Lesion)
2.Upload a medical image
3.Run AI analysis
4.Review predictions with confidence scores
5.Download diagnostic summary report

⚠️ Limitations

Models trained on limited academic datasets
Not validated for real clinical environments
Performance may vary across imaging devices and populations

⚕️ Medical Disclaimer

This project is intended for educational and research purposes only.
It is not a certified medical diagnostic tool and should not be used for clinical decision-making.

👨‍💻 Author

Bandi Akshitha
GitHub: https://github.com/bakshitha07
LinkedIn: https://linkedin.com/in/akshitha-b-135b32312
Email: bakshitha7@gmail.com


📜 License
This project is licensed under the MIT License.
⭐ Acknowledgments

TensorFlow and Keras teams
React and Tailwind CSS communities
Public medical imaging datasets
Research literature on transfer learning in medical imaging

