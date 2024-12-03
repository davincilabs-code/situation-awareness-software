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
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  
    return img

def prepare_image_onnx(image_path, input_size=(640, 640)):
    img = Image.open(image_path).convert('RGB')  # 항상 RGB로 변환하여 3채널을 보장
    img = img.resize(input_size) 
    img = np.array(img, dtype=np.float32) / 255.0  # 정규화 (H, W, C)
    img = np.transpose(img, (2, 0, 1))           # (H, W, C) -> (C, H, W)
    img = np.expand_dims(img, axis=0) 
    return img

def benchmark_inference_extended(model_tf_lite_16, model_tf_lite_32, model_onnx, image_paths, num_iterations=1):

    tf_lite_16_times, tf_lite_32_times, onnx_times = [], [], []
    tf_lite_16_memories, tf_lite_32_memories, onnx_memories = [], [], []
    
    # 현재 프로세스 정보 가져오기
    process = psutil.Process(os.getpid())

    if model_tf_lite_16:
        for image_path in tqdm(image_paths, desc="TensorFlow Lite Float16"):
            image = prepare_image_tflite(image_path)
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                model_tf_lite_16.set_tensor(model_tf_lite_16.get_input_details()[0]['index'], image)
                model_tf_lite_16.invoke()
                tf_lite_16_times.append(time.perf_counter() - start_time)
                memory_info = process.memory_info()
                tf_lite_16_memories.append(memory_info.rss)

    if model_tf_lite_32:
        for image_path in tqdm(image_paths, desc="TensorFlow Lite Float32"):
            image = prepare_image_tflite(image_path)
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                model_tf_lite_32.set_tensor(model_tf_lite_32.get_input_details()[0]['index'], image)
                model_tf_lite_32.invoke()