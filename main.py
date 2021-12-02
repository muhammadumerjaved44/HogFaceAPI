from os import path
from typing import Optional
from fastapi import FastAPI
from starlette.responses import JSONResponse
import read_images

app = FastAPI()

@app.get("/find_label")
def read_root(image_path:str = 'image_path/'):
    result = read_images.single_image_predict(image_path)
    return result

    

