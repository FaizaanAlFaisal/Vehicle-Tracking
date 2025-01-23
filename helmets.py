import cv2
from ultralytics import YOLOWorld, YOLO
from utils import RealTimeVideoCapture


def helmet_near_vehicle(vehicle_bbox:tuple, helmet_centroid:tuple, threshold:int=90):
    """
    Check if the helmet is near the vehicle

    Args:
        vehicle_bbox (tuple): Bounding box of the vehicle, (x1, y1, x2, y2)
        helmet_centroid (tuple): Centroid of the helmet, (x, y)
        threshold (int): Region around the vehicle to consider the helmet near the vehicle
    """
    vx_min, vy_min, vx_max, vy_max = vehicle_bbox
    h_centroid_x, h_centroid_y = helmet_centroid
    
    if vx_min <= h_centroid_x <= vx_max and vy_min <= h_centroid_y <= vy_max:
        return True
    
    if (vx_min - threshold <= h_centroid_x <= vx_max + threshold and
        vy_min - threshold <= h_centroid_y <= vy_max + threshold):
        return True

    return False



model = YOLOWorld(model="model/yolov8l-worldv2.pt")
model.set_classes(["helmet"])


vehicle_model = YOLOWorld(model="model/yolov8s-worldv2.pt")
vehicle_model.set_classes(["motorcycle", "bicycle"])


cap = RealTimeVideoCapture("data/vids/vid3.mp4", capped_fps=True, restart_on_end=True, framerate=10)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit(1)

cap.start()


while True:
    ret, frame = cap.read()

    if not ret:
        continue
    
    frame = cv2.resize(frame, (1280, 720))

    helmet_results = model.predict(frame, verbose=False, conf=0.6)
    vehicle_results = vehicle_model.track(frame, verbose=False)

    helmet_locations = []

    for res in helmet_results:
        for det in res.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            x, y, _, _ = map(int, det.xywh[0])
            helmet_locations.append((x, y))
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

    for v_res in vehicle_results:
        for det in v_res.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            
            helmet_detected = False
            if len(helmet_locations) > 0:
                for helmet_loc in helmet_locations:
                    if helmet_near_vehicle((x1, y1, x2, y2), helmet_loc):
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Wearing helmet", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        helmet_detected = True
                        break
            
            if not helmet_detected:
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "No Helmet", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
