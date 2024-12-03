import os
import time
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

class MyFramework:
    def __init__(self, model_path, device="cpu"):
        """
        프레임워크 초기화 및 모델 로드.
        :param model_path: 모델 파일 경로
        :param device: 실행 디바이스 ("cpu" 또는 "cuda")
        """
        self.model_path = model_path
        self.model = None
        self.session = None
        self.model_type = None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu") if device == "cuda" else torch.device("cpu")
        self.load_model()

    def load_model(self):
        """
        파일 확장자를 기반으로 모델 로드.
        """
        file_extension = Path(self.model_path).suffix.lower()

        if file_extension == ".pt":
            self.model_type = "pytorch"
            print("PyTorch 모델 로드 중...")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model = checkpoint['model'].to(self.device)  # 모델 객체 디바이스로 이동
            self.model.eval()

            if self.device.type == "cpu":
                self.model.float()  # 모델의 모든 파라미터를 float32로 변환
            print("PyTorch 모델 로드 완료.")

        elif file_extension == ".tflite":
            self.model_type = "tensorflow"
            print("TensorFlow Lite 모델 로드 중...")
            self.model = tf.lite.Interpreter(model_path=str(self.model_path))
            self.model.allocate_tensors()
            print("TensorFlow Lite 모델 로드 완료.")

        elif file_extension == ".onnx":
            self.model_type = "onnx"
            print("ONNX 모델 로드 중...")
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers and self.device.type == "cuda":
                providers = ["CUDAExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            print("ONNX 모델 로드 완료.")
        else:
            raise ValueError("지원되지 않는 파일 확장자입니다. ('pt', 'tflite', 'onnx' 중 선택)")

    def preprocess_image(self, image_path, input_size):
        """
        입력 이미지를 모델 타입에 맞게 전처리합니다.
        :param image_path: 입력 이미지 경로
        :param input_size: 모델 입력 크기 (width, height)
        """
        print(f"모델 타입: {self.model_type}")
        img = Image.open(image_path).convert('RGB')
        img = img.resize(input_size)
        img_array = np.array(img, dtype=np.float32) / 255.0

        if self.model_type == "pytorch":
            img_array = np.transpose(img_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)
            return img_tensor.to(self.device)

        elif self.model_type == "tensorflow":
            img_array = np.expand_dims(img_array, axis=0)  # (H, W, C) -> (1, H, W, C)
            return img_array

        elif self.model_type == "onnx":
            img_array = np.expand_dims(np.transpose(img_array, (2, 0, 1)), axis=0)  # (H, W, C) -> (1, C, H, W)
            return img_array.astype(np.float32)  # 데이터 타입을 float32로 변환
        else:
            raise ValueError("전처리를 지원하지 않는 모델 타입입니다.")
        
    def predict(self, image_path, input_size=(640, 640)):
        """
        이미지를 입력받아 추론 결과 반환.
        :param image_path: 입력 이미지 경로
        :param input_size: 모델 입력 크기 (width, height)
        """
        input_data = self.preprocess_image(image_path, input_size)

        if self.model_type == "pytorch":
            with torch.no_grad():
                if self.device.type == "cuda":
                    input_data = input_data.to(torch.float16).to(self.device)  # GPU에서 실행 시 float16
                else:
                    input_data = input_data.to(torch.float32).to(self.device)  # CPU에서 실행 시 float32
                start_time = time.time()
                output = self.model(input_data)
                inference_time = (time.time() - start_time) * 1000  # ms
                print(f"PyTorch 추론 완료! 소요 시간: {inference_time:.2f} ms")
                if isinstance(output, (tuple, list)):
                    output = [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in output]
                elif isinstance(output, torch.Tensor):
                    output = output.cpu().numpy()
                return output

        elif self.model_type == "tensorflow":
            input_index = self.model.get_input_details()[0]['index']
            output_index = self.model.get_output_details()[0]['index']
            self.model.set_tensor(input_index, input_data)
            start_time = time.time()
            self.model.invoke()
            output = self.model.get_tensor(output_index)
            inference_time = (time.time() - start_time) * 1000  # ms
            print(f"TensorFlow Lite 추론 완료! 소요 시간: {inference_time:.2f} ms")
            return output

        elif self.model_type == "onnx":
            input_name = self.session.get_inputs()[0].name
            start_time = time.time()
            output = self.session.run(None, {input_name: input_data})
            inference_time = (time.time() - start_time) * 1000  # ms
            print(f"ONNX 추론 완료! 소요 시간: {inference_time:.2f} ms")
            return output[0]

        else:
            raise ValueError("추론을 지원하지 않는 모델 타입입니다.")

# 사용 예시
if __name__ == "__main__":
    print("\n=== PyTorch 모델 ===")
    model = MyFramework("test.pt")
    output_pt = model.predict("test.jpg", input_size=(640, 640))

    print("\n=== TensorFlow Lite Float16 모델 ===")
    model = MyFramework("test_float16.tflite")
    output_tflite16 = model.predict("test.jpg", input_size=(640, 640))

    print("\n=== TensorFlow Lite Float32 모델 ===")
    model = MyFramework("test_float32.tflite")
    output_tflite32 = model.predict("test.jpg", input_size=(640, 640))

    print("\n=== ONNX 모델 ===")
    model = MyFramework("test.onnx")
    output_onnx = model.predict("test.jpg", input_size=(640, 640))