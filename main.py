import threading
from queue import Queue
from utils import RealTimeVideoCapture
from ultralytics import YOLO, YOLOWorld
from ultralytics.utils.plotting import save_one_box
from paddleocr import PaddleOCR
import numpy as np
import re
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
                    classes:list=["motorcycle", "car", "bus"], capped_fps:bool=True, restart_on_end:bool=True, framerate:int=30,
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

        # dictionary to map class ids to determined license plate numbers
        self.lp_dict : dict = {} # {vehicle_id:  (lp_num, lp_conf, cls_name, valid) }

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
            results = self.model.track(frame, verbose=False, stream=True, persist=True)
            # results = self.model.track(frame, stream=True)
            annotated_frame = frame

            for res in results:
                for det in res.boxes:
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    cls_name = id_to_class[int(det.cls)]

                    # assign colours based on class, car = purple, motorcycle = blue, car = white
                    if cls_name == "car":
                        color = (255, 255, 255)
                    elif cls_name == "motorcycle":
                        color = (255, 50, 25)
                    elif cls_name == "bus":
                        color = (128, 0, 128)
                    
                    # draw bounding box
                    annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # if no id, no need to run detection as we won't be able to do anything with lp results
                    if det.id is None:
                        continue

                    id = int(det.id[0])

                    vehicle_box = save_one_box(det.xyxy, res.orig_img, BGR=True, save=False)
                    self.vehicles_queue.put([[det.id, cls_name], vehicle_box])

                    if id in self.lp_dict:
                        lp_num, lp_conf, _, _ = self.lp_dict[id]
                        if lp_num == "" or lp_conf == 0.0:
                            lp_num = "Bad Angle"
                        cv2.putText(annotated_frame, f"{lp_num} - {lp_conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 4, lineType=cv2.LINE_AA)
                        cv2.putText(annotated_frame, f"{lp_num} - {lp_conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, lineType=cv2.LINE_AA)

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

            v_id, cls_name = info
            v_id = int(v_id[0])

            results = self.lp_model.predict(vehicle_box, verbose=False, stream=True)

            for res in results:
                for det in res.boxes:
                    xyxy = det.xyxy
                    conf = det.conf[0]
                    
                    # lp detection confidence threshold
                    if conf < 0.6:
                        continue
                    
                    cropped_lp = save_one_box(xyxy, vehicle_box.copy(), BGR=True, save=False)
                    lp_number, confidence, valid = self.license_plate_ocr(cropped_lp, cls_name)

                    # automatically update the dictionary if no entry exists
                    if v_id not in self.lp_dict:
                        self.lp_dict[v_id] = (lp_number, confidence, cls_name, valid)

                    # if an entry already exists, update it
                    else:
                        # if new read is valid and old read is invalid, update
                        if valid and not self.lp_dict[v_id][3]:
                            self.lp_dict[v_id] = (lp_number, confidence, cls_name, valid)
                            continue

                        # if new read is more confident than old read, update
                        if valid and confidence > self.lp_dict[v_id][1]:
                            self.lp_dict[v_id] = (lp_number, confidence, cls_name, valid)
            # print(self.lp_dict)


    def license_plate_ocr(self, plate_img:np.ndarray, class_name:str) -> tuple[str, float, bool]:
        """
        Extracts the license plate number from a cropped image.
        
        Args:
            plate_image (np.ndarray): The cropped image of the license plate.
            class_name (str): The class name of the vehicle.
            
        Returns:
            tuple: A tuple containing the extracted license plate number (str), the average confidence score (float) and a boolean value indicating if the license plate is valid.
        """

        # preprocess the image
        preprocessed_image = self.__lp_image_processing__(plate_img)
        
        # perform OCR
        lp_results = self.ocr.ocr(preprocessed_image, cls=True)
        # lp_results = self.ocr.ocr(binary_image, cls=True)

        license_plate_number = ""
        confidence_scores = []

        if len(lp_results) == 0:
            return "", 0.0, False
        
        for lp_res in lp_results:
            if lp_res is None:
                continue
            for line in lp_res:
                license_plate_number += line[1][0]
                confidence_scores.append(int(float(line[1][1]) * 100))
                # print(f"{txt} - {conf}", end="\n\n")

        valid, license_plate_number = self.apply_lp_ocr_rules(license_plate_number, class_name)
        average_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        # if average_confidence >= 80:
        #     filename = f"ocr_results/lp_{license_plate_number}.jpg"
        #     # cv2.imwrite(filename, plate_img)
        #     cv2.imwrite(filename, preprocessed_image)

        if license_plate_number == "":
            return "", 0.0, False
        return license_plate_number, average_confidence, valid
    

    def apply_lp_ocr_rules(self, license_plate:str, class_name:str) -> tuple[bool, str]:
        """
        Applies rules to the extracted license plate number to make it more readable.

        Args:
            license_plate (str): The extracted license plate number.
            class_name (str): The class name of the vehicle.

        Returns:
            tuple: A tuple containing a boolean value indicating if the license plate is valid and the modified license plate number.
        """
        text = license_plate.replace(' ', '')
        
        car_bus_pattern = r'^[A-Za-z]{3}\d{3}[A-Za-z]{1}$'  # 3 letters, 3 digits, 1 letter
        motorcycle_pattern = r'^[A-Za-z]{2}\d{3}[A-Za-z]{1}$'  # 2 letters, 3 digits, 1 letter

        if class_name == "car" or class_name == "bus":
            if re.match(car_bus_pattern, text):
                return True, text
            
        if class_name == "motorcycle":
            if re.match(motorcycle_pattern, text):
                return True, text
            
        return False, text


    def __lp_image_processing__(self, image:np.ndarray) -> np.ndarray:
        """
        Preprocesses the license plate image for better detection results.

        Args:
            image (np.ndarray): The license plate image.

        Returns:
            np.ndarray: The preprocessed image with raised contrast and readability.
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
    vt = VehicleTracking(video_source=src, framerate=5, resize=(1920, 1080))
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