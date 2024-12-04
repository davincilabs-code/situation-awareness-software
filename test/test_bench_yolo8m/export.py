import time
import os
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import requests
import zipfile

# 데이터셋 다운로드 및 경로 설정
dataset_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
dataset_path = "coco128"

def download_and_extract(url, dest_path):
    zip_path = "coco128.zip"
    if not os.path.exists(dest_path):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall()
        os.remove(zip_path)  # 압축 파일 삭제

download_and_extract(dataset_url, dataset_path)

model_path = "yolov8m.pt"

def load_models(model_path):
    tf_lite_model = YOLO(model_path)
    tf_lite_model.export(format='tflite')

    tflite_path = Path(model_path).with_suffix('.tflite')

    if not tflite_path.exists():
        print(f"Error: {tflite_path} not found. Please check the export process.")
        return None, None
    
    tf_lite_interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    tf_lite_interpreter.allocate_tensors()

    onnx_path = Path(model_path).with_suffix('.onnx')
    YOLO(model_path).export(format='onnx')
    if not onnx_path.exists():
        print(f"Error: {onnx_path} not found. Please check the export process.")
        return None, None
    
    ort_sess = ort.InferenceSession(str(onnx_path), providers=['CUDAExecutionProvider'])

    return tf_lite_interpreter, ort_sess

def main():
    image_folder = Path(dataset_path) / "images" / "train"
    image_paths = [str(p) for p in image_folder.glob("*.jpg")]

    print("모델 로딩 중...")
    tf_lite_model, onnx_model = load_models(model_path)

    if tf_lite_model is None or onnx_model is None:
        print("모델 로드에 실패했습니다. 필요한 라이브러리가 모두 설치되었는지 확인하세요.")
        return

if __name__ == "__main__":
    main()