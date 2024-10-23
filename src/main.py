import cv2
from detect_targets import TargetDetector, process_image
import config

def process_video(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect(frame)
        frame_with_boxes = detector.draw_boxes(frame, results)
        
        cv2.imshow('Detection', frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    detector = TargetDetector(config.MODEL_PATH, config.CONF_THRESHOLD, config.IOU_THRESHOLD)

    # 处理图片
    if config.PROCESS_IMAGE:
        print("Processing image...")
        image_with_boxes = process_image(config.IMAGE_PATH, detector)
        cv2.imshow('Image Detection', image_with_boxes)
        print("Press 'q' to quit image detection and move to video detection (if enabled)...")
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    # 处理视频
    if config.PROCESS_VIDEO:
        print("Processing video...")
        process_video(config.VIDEO_PATH, detector)

    print("Detection completed.")

if __name__ == "__main__":
    main()
