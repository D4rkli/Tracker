import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from matplotlib import pyplot as plt
from yolo_v5 import dict_object


class VideoProcessor:
    def __init__(self, video_path, yolo_model):
        self.video_capture = cv2.VideoCapture(video_path)
        self.yolo_model = yolo_model
        self.deepsort = DeepSort()

        plt.ion()
        self.fig, self.ax = plt.subplots()

        self.window_name = "Video Stream"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def process(self):
        if not self.video_capture.isOpened():
            print("Error: Unable to open video")
            return

        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break

            boxes, confidences, classes = self.yolo_model.predict(frame)

            bbs = []

            color_white = (255, 255, 255)
            color_yellow = (0, 255, 255)

            for j in range(len(boxes)):
                b = boxes[j]
                x1 = int(b[0])
                y1 = int(b[1])
                x2 = int((b[2] + b[0]))
                y2 = int((b[3] + b[1]))

                vehicle = dict_object[classes[j]]
                rect_top_left = (x1, y1)
                rect_bottom_right = (x2, y2)
                text_top_left = (x1, y1 - 10)

                thickness = 1
                shift = 0
                font_scale = 0.5

                cv2.rectangle(frame, rect_top_left, rect_bottom_right, color_white, thickness, cv2.LINE_8, shift)
                cv2.putText(frame, vehicle, text_top_left, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_yellow,
                            thickness)
                bbs.append(([x1, y1, x2, y2], confidences[j], vehicle))

            emb = self.deepsort.generate_embeds(frame, bbs)

            tracks = self.deepsort.update_tracks(bbs, embeds=emb, frame=frame)

            for k, track in enumerate(tracks):
                if not track.is_confirmed():
                    continue
                if track.original_ltwh is None:
                    continue
                cv2.putText(frame, track.track_id, (int(track.original_ltwh[0]), int(track.original_ltwh[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color_yellow, 1)

            # Отображение кадра с треками
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Отображение текущего кадра
            cv2.imshow(self.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()
        plt.ioff()
