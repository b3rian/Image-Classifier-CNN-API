# 🧠 Image Classifier API & UI — ResNet50 & EfficientNetV2L

This project is a full-stack deep learning deployment pipeline for image classification using two powerful models: ResNet50 and EfficientNetV2L. It showcases the training of high-performance models, REST API development using FastAPI, and a Streamlit frontend that allows users to interact with the models through a user-friendly web interface. The project is containerized with Docker and CI/CD automated using GitHub Actions. It is deployed to production via Hugging Face Spaces and Streamlit Community Cloud.

# 🚀 Project Features

✅ Trained **ResNet50** and **EfficientNetV2L** on ImageNet using Tensorflow distributed training on 2 GPUs.

✅ Achieved **top-1 accuracy** of:

ResNet50: **80%**, EfficientNetV2L: **86%**

✅ Achieved **top-5 accuracy** of:

ResNet50: **90%**, EfficientNetV2L: **96%**

✅ Built a **FastAPI** backend to serve model predictions.

✅ Built a **Streamlit** frontend that:
  - Accepts user images.
  - Lets users choose between **ResNet50**, **EfficientNetV2L**, or both.

✅ Integrated the backend and frontend to work seamlessly.

✅ Dockerized both API and frontend.

✅ Automated CI/CD with **GitHub Actions.**

✅ Deployed:

   - **API** to Hugging Face Spaces
   - **Frontend** to Streamlit Community Cloud

✅ Achieves average response latency of **~2.5 seconds.**

# 🔧 Tech Stack

🐍 **Python** – Core programming language

🧠 **Tensorflow** – Model training

⚡ **FastAPI** – REST API

🌐 **Streamlit** – Frontend UI

🐳 **Docker** – Containerization

🔁 **GitHub Actions** – CI/CD automation

☁️ **HF Spaces & Streamlit Cloud** – Deployment

# 📦 Installation (Local)

**Clone the repo**

git clone https://github.com/b3rian/Image-Classifier-CNN-API

cd image-classifier-api

**Build & Run with Docker**

- FastAPI API

cd api_backend
docker build -t image-classifier-api .

docker run -p 8000:8000 image-classifier-api

- Streamlit Frontend

cd streamlit_app

docker build -t image-classifier-ui .

docker run -p 8501:8501 image-classifier-ui

