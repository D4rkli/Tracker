import torch
from video_processor import VideoProcessor
from yolo_v5 import YOLOv5

if __name__ == "__main__":
    video_path = 'cars.mp4'

    # Создание экземпляра класса YOLOv5
    yolo_model = YOLOv5(path='best_6.0.pt',
                        device=torch.device("cpu"))

    # Создание экземпляра класса VideoProcessor
    processor = VideoProcessor(video_path, yolo_model)
    processor.process()
