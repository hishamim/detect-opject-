from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
from ultralytics import YOLO
import cv2
import numpy as np

# from yolo_model import detect_objects
# from pydantic import BaseModel


app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    model = YOLO("yolov8n.pt")



UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Object Detection</h2>
            <form action="/detect" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(img)

    annotated = results[0].plot()

    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    cv2.imwrite(path, annotated)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": model.names[int(box.cls)],
            "confidence": float(box.conf)
        })

    return {
        "detections": detections,
        "image_url": f"/static/uploads/{filename}"
    }

app.mount("/static", StaticFiles(directory="static"), name="static")
