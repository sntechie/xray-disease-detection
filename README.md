# 🩺 X-ray Disease Detection System

## 📌 Description
This project uses Deep Learning to detect diseases like Pneumonia from chest X-ray images.

## 🚀 Features
- Upload X-ray image
- Detect Normal or Pneumonia
- Built with Streamlit

## 📁 Project Structure
app/
model/
sample_images/

## 🛠️ Requirements
- Python 3.x
- streamlit
- tensorflow
- numpy
- opencv-python

## 📦 Installation
```bash
pip install -r requirements.txt
```

## ▶️ Run the Project
```bash
streamlit run app/streamlit_app.py
```

## 🧪 Testing
Use images from `sample_images/` folder to test the model.

## 📂 Dataset

Dataset is not included in this repository due to large size.

You can download the dataset from Kaggle:
https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images

After downloading, place it in the project folder like this:

dataset/
├── train/
└── test/
