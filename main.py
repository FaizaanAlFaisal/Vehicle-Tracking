import threading
from queue import Queue
from utils import RealTimeVideoCapture
from ultralytics import YOLO, YOLOWorld
from ultralytics.utils.plotting import save_one_box
from paddleocr import PaddleOCR
import cv2
import time


class VehicleTracking:
    """
    class VehicleTracking
    ----------------------
    This class is designed to track vehicles in a video feed. Primary function is detecting and reading license plates.

    Secondary functions include vehicle counting, helmet detection, etc.

    Parameters:
        video_source (str): Path to a video file or a video feed URL (rtsp/rtmp/http).
        model (str): Path to the YOLO model file for vehicle/helmet detection.
        lp_model (str): Path to the YOLO model file for license plate detection.
        classes (list): List of classes to detect in the model.
    """
    def __init__(
                    self, video_source:str, model:str="model/yolov8s-worldv2.pt", lp_model:str="model/yolo-license-plates.pt", 
                    classes:list=["helmet", "motorcycle"], capped_fps:bool=True, restart_on_end:bool=True, framerate:int=30,
                    resize:tuple=(1280, 720),
                ):
        print("Initialization Started.")
        # video capture
        self.video_source : str = video_source
        self.capped_fps : bool = capped_fps
        self.restart_on_end : bool = restart_on_end
        self.framerate : int = framerate
        self.resize : tuple = resize
        self.cap = RealTimeVideoCapture(video_source, capped_fps=capped_fps, restart_on_end=restart_on_end, framerate=framerate)
        print("Capture created.")
        
        # model initialization
        self.model : YOLOWorld = YOLOWorld(model=model)
        self.classes : list = classes
        self.model.set_classes(classes)
        print("Vehicle Detection model loaded successfully.")
        
        self.lp_model : YOLO = YOLO(model=lp_model)
        print("License Plate Detection model loaded successfully.")

        self.ocr : PaddleOCR = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        print("OCR Model loaded successfully.")

        # threading
        self.stop_event : threading.Event = threading.Event()
        self.vehicles_queue : Queue = Queue(maxsize=10) # putting vehicle bounding boxes in this queue to check lps, optimize by running sep thread

        self.tracking_thread : threading.Thread = threading.Thread(target=self.__tracking_thread__)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        print("Tracking thread started.")

        self.lp_thread : threading.Thread = threading.Thread(target=self.__license_plate_thread__)
        self.lp_thread.daemon = True
        self.lp_thread.start()
        print("License Plate thread started.")



    def __tracking_thread__(self):
        """
        Continuously reads frames from the video source and processes them for vehicle tracking.
        """
        self.cap.start()
        id_to_class = {i: item for i, item in enumerate(self.classes)}

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, self.resize)

            # detect vehicles and helmets
            results = self.model.predict(frame, verbose=False, stream=True)
            annotated_frame = frame

            for res in results:
                for det in res.boxes:
                    cls_name = id_to_class[int(det.cls)]
                    if cls_name in ['car', 'motorcycle', 'bus']:
                    # if int(det.cls) != 0: # if not helmet
                        vehicle_box = save_one_box(det.xyxy, res.orig_img, BGR=True, save=False)
                        
                        self.vehicles_queue.put(vehicle_box)
                annotated_frame = res.plot(conf=False)

            cv2.imshow("Vehicle Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    
    def __license_plate_thread__(self):
        """
        Wait for cropped images of vehicles, detect and read license plates from those images.
        """

        while not self.stop_event.is_set():
            vehicle_box = None
            try:
                vehicle_box = self.vehicles_queue.get(timeout=0.05)
            except:
                continue

            results = self.lp_model.predict(vehicle_box, verbose=False, stream=True)

            for res in results:
                for det in res.boxes:
                    xyxy = det.xyxy
                    cropped_lp = save_one_box(xyxy, res.orig_img, BGR=True, save=False)

                    lp_results = self.ocr.ocr(cropped_lp, cls=True)
                    if len(lp_results) == 0:
                        continue
                    for res in lp_results:
                        if res is None:
                            continue
                        for line in res:
                            txt = line[1][0]
                            conf = line[1][1]
                            print(f"{txt} - {conf}")
                    print()


    def stop(self):
        """
        Stops the tracking thread.
        """
        self.stop_event.set()
        self.cap.release()
        self.tracking_thread.join(2)
        self.lp_thread.join(2)




# driver code
def main():
    src = "data/vids/vid1.mp4"
    vt = VehicleTracking(video_source=src, classes=["helmet", "motorcycle", "car", "bus"], framerate=10)

    while True:
        time.sleep(0.5)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vt.stop()



if __name__ == "__main__":
    main()
