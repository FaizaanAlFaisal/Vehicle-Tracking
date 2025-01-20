import threading
from queue import Queue
from utils import RealTimeVideoCapture
from ultralytics import YOLO, YOLOWorld
from ultralytics.utils.plotting import save_one_box
from paddleocr import PaddleOCR
import numpy as np
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
                    self, video_source:str, model:str="model/yolov8s-world.pt", lp_model:str="model/yolo-license-plates.pt", 
                    classes:list=["helmet", "motorcycle"], capped_fps:bool=True, restart_on_end:bool=True, framerate:int=30,
                    resize:tuple=(1280, 720),
                ):
        print("Initialization Started.")

        # model initialization - done before any threads are created for performance
        self.model : YOLOWorld = YOLOWorld(model=model)
        self.classes : list = classes
        self.model.set_classes(classes)
        print("Vehicle Detection model loaded successfully.")
        
        self.lp_model : YOLO = YOLO(model=lp_model)
        print("License Plate Detection model loaded successfully.")

        self.ocr : PaddleOCR = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        self.ocr_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        print("OCR Model loaded successfully.")

        # video capture
        self.video_source : str = video_source
        self.capped_fps : bool = capped_fps
        self.restart_on_end : bool = restart_on_end
        self.framerate : int = framerate
        self.resize : tuple = resize
        self.cap = RealTimeVideoCapture(video_source, capped_fps=capped_fps, restart_on_end=restart_on_end, framerate=framerate)
        print("Capture created.")

        # threading
        self.stop_event : threading.Event = threading.Event()
        self.vehicles_queue : Queue = Queue(maxsize=10) # putting vehicle bounding boxes in this queue to check lps, optimize by running sep thread

        self.tracking_thread : threading.Thread = threading.Thread(target=self.__tracking_thread__)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        print("Tracking thread started.")

        self.lp_thread : threading.Thread = threading.Thread(target=self.__license_plate_detection_thread__)
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
            results = self.model.track(frame, verbose=False, stream=True)
            # results = self.model.track(frame, stream=True)
            annotated_frame = frame

            for res in results:
                for det in res.boxes:
                    cls_name = id_to_class[int(det.cls)]
                    id = det.id
                    if cls_name in ['car', 'motorcycle', 'bus']:
                    # if int(det.cls) != 0: # if not helmet
                        vehicle_box = save_one_box(det.xyxy, res.orig_img, BGR=True, save=False)

                        file_path = f"license_plates/vehicle_{id}.jpg"
                        cv2.imwrite(file_path, vehicle_box)
                        
                        self.vehicles_queue.put([[det.id, det.conf, det.xyxy], vehicle_box])
                annotated_frame = res.plot(conf=False)

            cv2.imshow("Vehicle Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    
    def __license_plate_detection_thread__(self):
        """
        Wait for cropped images of vehicles, detect license plates from those images.
        """

        while not self.stop_event.is_set():
            vehicle_box = None
            info = None
            try:
                info, vehicle_box = self.vehicles_queue.get(timeout=0.05)
            except:
                continue
            
            if vehicle_box is None:
                continue

            vehicle_id, vehicle_conf, vehicle_xyxy = info
            results = self.lp_model.predict(vehicle_box, verbose=False, stream=True)

            for res in results:
                for det in res.boxes:
                    xyxy = det.xyxy
                    conf = det.conf[0]

                    if conf < 0.6:
                        continue
                    cropped_lp = save_one_box(xyxy, vehicle_box.copy(), BGR=True, save=False)

                    lp_number, confidence = self.license_plate_ocr(cropped_lp)
                    if confidence < 80:
                        continue
                    print(f"License Plate: {lp_number}, OCR Confidence: {confidence}, LP Confidence: {conf}")
        

    def license_plate_ocr(self, plate_img:np.ndarray) -> tuple[str, float]:
        """
        Extracts the license plate number from a cropped image.
        
        Args:
            plate_image (np.ndarray): The cropped image of the license plate.
            
        Returns:
            tuple: A tuple containing the extracted license plate number (str) and the average confidence score (float).
        """

        # preprocess the image
        preprocessed_image = self.__lp_image_processing__(plate_img)
        
        # perform OCR
        lp_results = self.ocr.ocr(preprocessed_image, cls=True)
        # lp_results = self.ocr.ocr(binary_image, cls=True)

        license_plate_number = ""
        confidence_scores = []

        if len(lp_results) == 0:
            return "", 0.0
        
        for lp_res in lp_results:
            if lp_res is None:
                continue
            for line in lp_res:
                license_plate_number += line[1][0] + " "
                confidence_scores.append(int(float(line[1][1]) * 100))
                # print(f"{txt} - {conf}", end="\n\n")

        license_plate_number = license_plate_number.strip()
        average_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        if average_confidence >= 80:
            filename = f"ocr_results/lp_{license_plate_number}.jpg"
            # cv2.imwrite(filename, plate_img)
            cv2.imwrite(filename, preprocessed_image)


        if license_plate_number == "":
            return "", 0.0
        return license_plate_number, average_confidence


    def __lp_image_processing__(self, image:np.ndarray) -> np.ndarray:
        """
        Preprocesses the license plate image for better detection results.
        """
        # convert to grayscale
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # double the size
        processed_img = cv2.resize(processed_img, (2*processed_img.shape[1], 2*processed_img.shape[0]))
        # apply CLAHE
        processed_img = self.ocr_clahe.apply(processed_img)
        # sharpen
        processed_img = self.unsharp_mask(processed_img)
        return processed_img


    def unsharp_mask(self, image:np.ndarray, kernel_size:int=(5, 5), sigma:float=1.0, amount:int=0.5) -> np.ndarray:
        """
        unsharp_mask
        ------------
        Sharpens an image using the unsharp mask technique.
        """
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return sharpened

    def stop(self):
        """
        Stops the tracking thread.
        """
        self.stop_event.set()
        self.cap.release()
        self.tracking_thread.join(1)
        self.lp_thread.join(1)


# driver code
def main():
    src = "data/vids/vid1.mp4"
    vt = VehicleTracking(video_source=src, classes=["bicycle", "motorcycle", "car", "bus"], framerate=5)
    try:
        while True:
            time.sleep(0.5)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                vt.stop()
                break
    except KeyboardInterrupt:
        vt.stop()
        print("Stopping background threads.")


if __name__ == "__main__":
    main()