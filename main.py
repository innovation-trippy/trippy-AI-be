from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from typing import List
import uvicorn

from ultralytics import YOLO
import cv2

app = FastAPI()

@app.get("/")
def root():
    return {"meesage" : "Hello World!"}

async def parse_body(request: Request):
    data: bytes = await request.body()
    return data

@app.get("/test")
def AI_model_interface():
    image = cv2.imread('./img_example/img_example.jpeg', 1)
    
    model = YOLO('./model/yolov8m.pt')

    results = model([image], stream=False)

    img = results[0].plot()
    resized_img = cv2.resize(img, (600,400))

    cv2.imshow("img", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    boxes = results[0].boxes
    class_id = boxes.cls.cpu().detach().numpy().tolist()
    conf = boxes.conf.cpu().detach().numpy().tolist()

    detected_object = {
        "class_id": class_id,
        "conf" : conf
    }
    return detected_object

@app.post("/uploadimg")
async def AI_model_interface(files: List[UploadFile] = File(...)):
    model = YOLO('./model/yolov8m.pt')
    class_id = []
    conf = []

    for file in files:
        content = await file.read()

        if file.content_type not in ['image/jpeg', 'image/png']:
            raise HTTPException(status_code=406, detail="Please upload only .jpeg files")

        with open('./img_example/user_input.jpeg', 'wb') as f:
            f.write(content)

        image = cv2.imread('./img_example/user_input.jpeg', 1)

        results = model([image], stream=False)
 
        boxes = results[0].boxes
        new_id = boxes.cls.cpu().detach().numpy().tolist()
        new_conf = boxes.conf.cpu().detach().numpy().tolist()

        class_id = class_id + new_id
        conf = conf + new_conf

    detected_objects = {
        "class_id" : class_id,
        "conf": conf
    }
    return detected_objects

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)