from fastapi import FastAPI
from src import metrics
from tensorflow.keras.models import load_model

app = FastAPI(title="MyApp", description="Design Autonomous Car")


@app.get("/")
def read_root():
    return {"Hello": "World"}


# Load vgg16_model manually
vgg16_model = load_model('./../models/vgg16.keras',
                         custom_objects={
                                         'dice_coeff': metrics.dice_coeff,
                                         'dice_loss': metrics.dice_loss,
                                         'total_loss': metrics.total_loss,
                                         'jaccard': metrics.jaccard
                        })


