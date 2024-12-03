import time
import os
import numpy as np
import onnxruntime as ort
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import tracemalloc  # 메모리 추적
from statistics import mean, stdev
import matplotlib.pyplot as plt
import psutil

# 모델 경로 설정
save_dir = Path(r"/home/jetson/Downloads/yolov/yolov8m/yolov8m_saved_model/")
tflite_float16_path = save_dir / "yolov8m_float16.tflite"
tflite_float32_path = save_dir / "yolov8m_float32.tflite"
onnx_model_path = "yolov8m.onnx"


# 데이터셋 경로 설정
dataset_path = "coco128/images/train2017"

def load_interpreters():
    # TFLite Interpreter 설정 (float16 및 float32)
    tf_lite_interpreter_16 = tf.lite.Interpreter(model_path=str(tflite_float16_path))
    tf_lite_interpreter_16.allocate_tensors()

    tf_lite_interpreter_32 = tf.lite.Interpreter(model_path=str(tflite_float32_path))
    tf_lite_interpreter_32.allocate_tensors()

    ort_sess = ort.InferenceSession(str(onnx_model_path))

    return tf_lite_interpreter_16, tf_lite_interpreter_32, ort_sess

def prepare_image_tflite(image_path, input_size=(640, 640)):
    img = Image.open(image_path).convert('RGB')  # 이미지가 RGB로 변환되었는지 확인
    img = img.resize(input_size)    