# Design an autonomous car

Complete the 'Image Segmentation' part of an embedded computer vision system for autonomous vehicles.

---

## Technologies
- Python
- Tensorflow
- Keras
- FastApi
- Streamlit
- AWS EC2
- Git

## Deep Learning models 
- Unet mini
- VGG16
- VGG16 + Data augmentation

## Metrics to evaluate the models
- dice_coeff
- jaccard
- accuracy

---

## Clone repository and install dependancies

```sh
git clone git@github.com:ahmedaao/design-autonomous-car.git
cd design-autonomous-car
pip install -r requirements.txt
pip install . # Install modules from package src/
```
---

## Data Source

Input Images and output masks are dowloaded here: [CITYSCAPES](https://www.cityscapes-dataset.com/dataset-overview/)  
You have to dowload 2 zip files:    
1. P8_Cityscapes_leftImg8bit_trainvaltest.zip for original images
2. P8_Cityscapes_gtFine_trainvaltest.zip for masks

You have to manually unzip these 2 zip files:

```sh
mkdir -p data/raw
unzip P8_Cityscapes_leftImg8bit_trainvaltest.zip -d /path/to/data/raw
unzip P8_Cityscapes_gtFine_trainvaltest.zip -d /path/to/data/raw
```
---

## Run streamlit and fastAPI manually

1. **fastapi**:
  - Navigate to backend directory
  ```sh
  cd backend
  ```
  - Run the app
  ```sh
  uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
  ``` 
1. **streamlit**:
  - Navigate to frontend directory
  ```sh
  cd frontend
  ```
  - Run the app
  ```sh
  streamlit run streamlit_app.py
  ``` 

---

## Run streamlit and fastAPI with docker-compose

To facilitate the deployment and launch of our application, it is more advantageous to create Docker images for FastAPI and Streamlit, 
and then create associated containers using Docker Compose. 
Navigate to the project directory:
```sh
sudo docker-compose up -d
```
---

## Access to the application

To access the frontend streamlit of the app: 

http://127.0.0.1:8501 

To access the fastapi swagger of the app: 

http://127.0.0.1:8000/docs
