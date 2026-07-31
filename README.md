# 🧠 Brain Tumor Detection System

An advanced deep learning-based web application that detects and classifies brain tumors from MRI scans using state-of-the-art convolutional neural networks.

## 🚀 Features
- **Multiple AI Models:** Utilizes highly accurate **EfficientNetV2** and **InceptionV3** models.
- **NeuroEnsemble™ Mode:** Combines predictions from multiple models for maximum accuracy.
- **Modern User Interface:** A sleek, glassmorphism-inspired dark mode UI.
- **Instant Analysis:** Fast processing and real-time visual feedback with confidence scores.

## 📊 Dataset & Accuracy
The models were trained on the **Cranial MRI Scan Archive**, classifying images into 4 categories:
1. Glioma Tumor
2. Meningioma Tumor
3. Pituitary Tumor
4. No Tumor

- **InceptionV3 Accuracy:** ~85.50%
- **EfficientNetV2 Accuracy:** ~84.44%
*(Models were trained using Data Augmentation and Fine-tuning strategies for enhanced robustness).*

## 🖼️ User Interface
*(You can add a screenshot of your running website here by dragging and dropping an image while editing this file on GitHub!)*

## 🛠️ How to Run Locally

If you want to run this project on your local machine, follow these steps:

### Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

### Installation & Setup

1. **Clone the repository and switch to the branch:**
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   git checkout Sazid
   ```

2. **Activate Virtual Environment:**
   - **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
   - **Windows (CMD):** `venv\Scripts\activate`
   - **Mac/Linux:** `source venv/bin/activate`

3. **Install Dependencies:**
   *(Ensure you install required packages like TensorFlow, Flask, NumPy, Werkzeug, etc. if you are setting up a fresh environment).*
   ```bash
   pip install flask tensorflow numpy werkzeug pillow
   ```

4. **Run the Application:**
   ```bash
   python main.py
   ```

5. **Open in Browser:**
   Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 👨‍💻 Developer
**Modabbir Hasan Sazid**
