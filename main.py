import argparse
import torch
from video_processor import VideoProcessor
from yolo_v5 import YOLOv5


def main(args):
    video_path = args.video_path
    model_path = args.model_path

    # Создание экземпляра класса YOLOv5
    yolo_model = YOLOv5(path=model_path,
                        device=torch.device("cpu"))

    # Создание экземпляра класса VideoProcessor
    processor = VideoProcessor(video_path, yolo_model)
    processor.process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video process using YOLOc5")
    parser.add_argument("--video_path", type=str, default="cars.mp4", help="File path")
    parser.add_argument("--model_path", type=str, default='best_6.0.pt', help="YOLOv5 path")
    args = parser.parse_args()
    main(args)
