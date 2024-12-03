#     print("\n=== 벤치마크 결과 ===")
#     # 성능 측정 출력 (시간)
#     if tf_lite_16_times:
#         print(f"TensorFlow Lite Float16 평균 추론 시간: {mean(tf_lite_16_times[10:]) * 1000:.2f}ms")
#         print(f"TensorFlow Lite Float16 표준 편차: {stdev(tf_lite_16_times[10:]) * 1000:.2f}ms")
#     if tf_lite_32_times:
#         print(f"TensorFlow Lite Float32 평균 추론 시간: {mean(tf_lite_32_times[10:]) * 1000:.2f}ms")
#         print(f"TensorFlow Lite Float32 표준 편차: {stdev(tf_lite_32_times[10:]) * 1000:.2f}ms")
#     if onnx_times:
#         print(f"ONNX 평균 추론 시간: {mean(onnx_times[10:]) * 1000:.2f}ms")
#         print(f"ONNX 표준 편차: {stdev(onnx_times[10:]) * 1000:.2f}ms")
    
#     # 메모리 사용 출력
#     if tf_lite_16_memories:
#         print(f"TensorFlow Lite Float16 평균 메모리 사용량: {mean(tf_lite_16_memories[10:]) / 1024:.2f} KB")
#     if tf_lite_32_memories:
#         print(f"TensorFlow Lite Float32 평균 메모리 사용량: {mean(tf_lite_32_memories[10:]) / 1024:.2f} KB")
#     if onnx_memories:
#         print(f"ONNX 평균 메모리 사용량: {mean(onnx_memories[10:]) / 1024:.2f} KB")
    
#     # 벤치마크 그래프 저장
#     save_benchmark_graphs(tf_lite_16_times, tf_lite_32_times, onnx_times, 
#                           tf_lite_16_memories, tf_lite_32_memories, onnx_memories)

# if __name__ == "__main__":
#     main()


import time
import os
import numpy as np
import onnxruntime as ort
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
import psutil
from statistics import mean, stdev
import matplotlib.pyplot as plt

# 모델 경로 설정
save_dir = Path("/home/jetson/Downloads/yolov/yolov8n/yolov8n_saved_model/")
tflite_float16_path = save_dir / "yolov8n_float16.tflite"
tflite_float32_path = save_dir / "yolov8n_float32.tflite"
onnx_model_path = "yolov8n.onnx"
pt_model_path = "yolov8n.pt"

# 데이터셋 경로 설정
dataset_path = "coco128"

def load_models():
    """YOLOv8, TFLite, ONNX 모델을 로드합니다."""
    # YOLOv8 PyTorch 모델
    yolo_model = YOLO(pt_model_path)
    yolo_model.to('cpu')  # CPU로 이동

    # TFLite 모델 로드
    tf_lite_interpreter_16 = tf.lite.Interpreter(model_path=str(tflite_float16_path))
    tf_lite_interpreter_16.allocate_tensors()
    
    tf_lite_interpreter_32 = tf.lite.Interpreter(model_path=str(tflite_float32_path))
    tf_lite_interpreter_32.allocate_tensors()

    ort_sess = ort.InferenceSession(str(onnx_model_path))  # providers 매개변수 제거

    # # CUDA를 우선적으로 사용
    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']


    # # ONNX 모델 로드
    # ort_sess = ort.InferenceSession(
    # str(onnx_model_path),
    # providers=providers
    # )

    return yolo_model, tf_lite_interpreter_16, tf_lite_interpreter_32, ort_sess

def prepare_image(image_path, input_size=(640, 640)):
    """이미지를 모델 입력 형식에 맞게 준비."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size)
    img = np.array(img, dtype=np.float32) / 255.0  # 정규화
    return img

def prepare_image_tflite(image, input_details):
    """TensorFlow Lite 입력 데이터 변환."""
    img = np.expand_dims(image, axis=0)
    return img.astype(np.float32)

def benchmark_inference(models, image_paths, num_iterations=1):
    """모델별 추론 속도와 메모리 사용량을 측정합니다."""
    times, memories = {}, {}
    process = psutil.Process(os.getpid())

    for model_name, model in models.items():
        model_times, model_memories = [], []
        print(f"Running {model_name} inference...")
        
        for image_path in tqdm(image_paths, desc=model_name):
            img = prepare_image(image_path)
            
            if "TFLite" in model_name:
                interpreter = model
                input_details = interpreter.get_input_details()
                interpreter.set_tensor(input_details[0]['index'], prepare_image_tflite(img, input_details))
                
                for _ in range(num_iterations):
                    start_time = time.perf_counter()
                    interpreter.invoke()
                    model_times.append(time.perf_counter() - start_time)
                    model_memories.append(process.memory_info().rss)
            
            elif model_name == "ONNX":
                input_name = model.get_inputs()[0].name
                img_onnx = np.transpose(img, (2, 0, 1))
                img_onnx = np.expand_dims(img_onnx, axis=0)

                for _ in range(num_iterations):
                    start_time = time.perf_counter()
                    model.run(None, {input_name: img_onnx})
                    model_times.append(time.perf_counter() - start_time)
                    model_memories.append(process.memory_info().rss)

            elif model_name == "pytorch":
                for _ in range(num_iterations):
                    start_time = time.perf_counter()
                    model.predict(img, save=False, verbose=False, cache=False)
                    model_times.append(time.perf_counter() - start_time)
                    model_memories.append(process.memory_info().rss)

        times[model_name] = model_times
        memories[model_name] = model_memories

    return times, memories

# def calculate_map(models, dataset_path):
#     """각 모델에 대한 mAP 계산."""
#     maps = {}
#     for model_name, model in models.items():
#         if model_name == "YOLOv8 (.pt)":
#             print(f"Calculating mAP for {model_name}...")
#             results = model.val(data=dataset_path)
#             maps[model_name] = (results.box.map50, results.box.map)
#         else:
#             # TFLite 및 ONNX의 mAP 측정을 위해서는 맞춤 구현 필요
#             # 추론 결과를 기반으로 직접 계산하는 로직 추가
#             print(f"mAP calculation not directly supported for {model_name}.")
#             maps[model_name] = (None, None)

#     return maps

def save_benchmark_results(times, memories):
    """추론 속도, 메모리 사용량 및 mAP 점수 저장."""
    model_names = list(times.keys())
    avg_times = [mean(times[model_name]) * 1000 for model_name in model_names]
    avg_memories = [mean(memories[model_name]) / 1024 for model_name in model_names]
    # map_scores = [maps[model_name][0] if maps[model_name][0] else 0 for model_name in model_names]

    # 추론 시간 그래프
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, avg_times, color=['blue', 'orange', 'green', 'red'])
    plt.title("Average Inference Time (ms)")
    plt.ylabel("Time (ms)")
    plt.savefig("inference_times.png")
    plt.close()

    # 메모리 사용량 그래프
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, avg_memories, color=['blue', 'orange', 'green', 'red'])
    plt.title("Average Memory Usage (KB)")
    plt.ylabel("Memory (KB)")
    plt.savefig("memory_usage.png")
    plt.close()

    # # mAP 점수 그래프
    # plt.figure(figsize=(10, 6))
    # plt.bar(model_names, map_scores, color=['blue', 'orange', 'green', 'red'])
    # plt.title("mAP50 Scores")
    # plt.ylabel("mAP50")
    # plt.savefig("map_scores.png")
    # plt.close()

def main():
    print("모델 로드 중...")
    yolo_model, tf_lite_16, tf_lite_32, onnx_model = load_models()

    models = {
        "pytorch": yolo_model,
        "TFLite Float16": tf_lite_16,
        "TFLite Float32": tf_lite_32,
        "ONNX": onnx_model,
    }

    print("이미지 파일 로드 중...")
    image_folder = Path(dataset_path) / "images/train2017"
    image_paths = [str(p) for p in image_folder.glob("*.jpg")]

    print("추론 벤치마크 시작...")
    times, memories = benchmark_inference(models, image_paths)

    # print("mAP 계산 중...")
    # maps = calculate_map(models, dataset_path)

    print("\n=== 결과 ===")
    for model_name in models.keys():
        print(f"{model_name}:")
        print(f"  - 평균 추론 시간: {mean(times[model_name]) * 1000:.2f} ms")
        print(f"  - 평균 메모리 사용량: {mean(memories[model_name]) / 1024:.2f} KB")
        # if maps[model_name][0]:
        #     print(f"  - mAP50: {maps[model_name][0]:.4f}, mAP50-95: {maps[model_name][1]:.4f}")
        # else:
        #     print(f"  - mAP: 지원되지 않음")

    save_benchmark_results(times, memories)

if __name__ == "__main__":
    main()
