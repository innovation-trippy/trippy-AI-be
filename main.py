from fastapi import FastAPI, HTTPException, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from typing import List, Set
from dotenv import load_dotenv
import pandas as pd
import json

from ultralytics import YOLO
import cv2
import numpy as np

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()

scanning_clients: Set[WebSocket] = set()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('./model/train19.pt')

llm1 = ChatOpenAI(model="gpt-4", temperature=1)
df1 = pd.read_csv("./csv/Forbid Item Table.csv")
df2 = pd.read_csv("./csv/special clause.csv")
df3 = pd.read_csv("./csv/comment.csv")
agent = create_pandas_dataframe_agent(llm1, [df1, df2, df3], verbose=False, agent_type='openai-tools')


@app.get("/")
def root():
    return {"meesage" : "Hello World!"}

@app.post("/chatbot")
async def chatbot_interface(request: Request):
    try:
        body = await request.body()
        data = body.decode('utf-8')
        question = json.loads(data)
        if not question:
            return {'answer': '질문을 입력해주세요.'}
        try:
            result = agent.invoke(question + " 한글로 답변해줘.")
            answer = result['output']
            return answer 
        except:
            return {'answer': '입력을 조금 구체적으로 해주세요'}
    except:
        return {'answer': '입력 에러'}

@app.websocket("/uploadvideo")
async def forbid_scanning_interface(websocket: WebSocket): 

    await websocket.accept()
    scanning_clients.add(websocket)
    detected_classes = set()

    def checkNewClass(class_id):
        before = len(detected_classes)
        for id in class_id:
            detected_classes.add(model.names[id])
        after = len(detected_classes)
        if before != after:
            return ', '.join(detected_classes)
        return None

    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            results = model(img, stream=True)
            for result in results:
                boxes = result.boxes
                class_id = boxes.cls.cpu().detach().numpy().tolist()
                class_name = checkNewClass(class_id)
                if class_name:
                    await websocket.send_text(class_name)
                image = result.plot()
                _, image_bytes = cv2.imencode('.jpeg', image)
                await websocket.send_bytes(image_bytes.tobytes())

    except WebSocketDisconnect:
        print('socket disconnected')
        scanning_clients.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)