import base64
import hashlib
import time
import json

from io import BytesIO
from typing import Dict

from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from funcaptcha_challenger import predict
from pydantic import BaseModel

from util.log import logger
from util.model_support_fetcher import ModelSupportFetcher

from pathlib import Path
MAP_FILE = Path("question_model_map.json")
UNMAPPED_QUESTIONS_FILE = Path("unmapped_questions.txt")

with MAP_FILE.open("r", encoding="utf-8") as f:
    QUESTION_MODEL_MAP = json.load(f)

def map_question_to_model(question: str) -> str | None:
    q = question.strip().lower()

    for entry in QUESTION_MODEL_MAP:
        keywords = entry["keywords"]
        model = entry["model"]
        if all(keyword in q for keyword in keywords):
            return model

    # Không map được → ghi log
    with UNMAPPED_QUESTIONS_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{question}\n")

    return None

app = FastAPI()
PORT = 8282
IS_DEBUG = True
fetcher = ModelSupportFetcher()

# In-memory task store
task_storage: Dict[str, dict] = {}


class Task(BaseModel):
    type: str
    image: str
    question: str


class TaskData(BaseModel):
    clientKey: str
    task: Task


def process_image(base64_image: str, variant: str):
    if base64_image.startswith("data:image/"):
        base64_image = base64_image.split(",")[1]

    image_bytes = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_bytes))

    ans = predict(image, variant)
    logger.debug(f"predict {variant} result: {ans}")
    return ans


@app.post("/createTask")
async def create_task(data: TaskData):
    client_key = data.clientKey
    task_type = data.task.type
    image = data.task.image
    question = data.task.question
    ans = {
        "errorId": 0,
        "errorCode": "",
        "status": "ready",
        "solution": {}
    }

    task_id = hashlib.md5(str(int(time.time() * 1000)).encode()).hexdigest()
    ans["taskId"] = task_id

    if question in fetcher.supported_models:
        ans["solution"]["objects"] = [process_image(image, question)]
    else:
        ans["errorId"] = 1
        ans["errorCode"] = "ERROR_TYPE_NOT_SUPPORTED"
        ans["status"] = "error"
        ans["solution"]["objects"] = []

    return ans


@app.post("/createRawTask")
async def create_raw_task(data: TaskData):
    client_key = data.clientKey
    task_type = data.task.type
    image = data.task.image
    question = data.task.question

    task_id = hashlib.md5(str(int(time.time() * 1000)).encode()).hexdigest()

    task_storage[task_id] = {
        "clientKey": client_key,
        "type": task_type,
        "image": image,
        "question": question,
        "status": "processing",
        "solution": {}
    }

    return {
        "errorId": 0,
        "errorCode": "",
        "status": "processing",
        "taskId": task_id
    }


@app.post("/getTaskResult")
async def get_task_result(payload: dict):
    task_id = payload.get("taskId")
    task = task_storage.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    print(f"\n[DEBUG] ⏳ Đang xử lý taskId: {task_id}")
    print(f"[DEBUG] ❓ Câu hỏi gửi lên: {task['question']}")
    print(f"[DEBUG] 📋 Danh sách model được hỗ trợ: {fetcher.supported_models}")

    if task["status"] == "ready":
        print(f"[DEBUG] ✅ Task {task_id} đã xử lý xong từ trước.")
        return {
            "errorId": 0,
            "errorCode": "",
            "status": "ready",
            "solution": task["solution"],
            "taskId": task_id
        }

    # Ánh xạ câu hỏi sang model name
    mapped_question = map_question_to_model(task["question"])
    print(f"[DEBUG] 🔁 Câu hỏi được ánh xạ thành model: {mapped_question}")

    if mapped_question and mapped_question in fetcher.supported_models:
        print(f"[DEBUG] ✅ Model được hỗ trợ. Đang xử lý ảnh...")
        result = process_image(task["image"], mapped_question)
        task["solution"]["objects"] = [result]
        task["status"] = "ready"
        print(f"[DEBUG] 🎯 Kết quả nhận dạng: {result}")
        return {
            "errorId": 0,
            "errorCode": "",
            "status": "ready",
            "solution": task["solution"],
            "taskId": task_id
        }
    else:
        print(f"[DEBUG] ❌ Không tìm thấy model phù hợp.")
        task["status"] = "error"
        task["solution"]["objects"] = []
        return {
            "errorId": 1,
            "errorCode": "ERROR_TYPE_NOT_SUPPORTED",
            "status": "error",
            "solution": task["solution"],
            "taskId": task_id
        }



@app.get("/support")
async def support():
    return fetcher.supported_models

@app.post("/getBalance")
async def get_balance(request: Request):
    return {
        "errorId": 0,
        "errorDescription": "",
        "balance": 999.99,
        "quantity": 99999
    }


@app.exception_handler(Exception)
async def error_handler(request: Request, exc: Exception):
    logger.error(f"error: {exc}")
    return {
        "errorId": 1,
        "errorCode": "ERROR_UNKNOWN",
        "status": "error",
        "solution": {"objects": []}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
