import cv2
import numpy as np

# --- Paths to YOLOv3 files ---
config_path = "D:/test/opencv/yolov3.cfg"
weights_path = "D:/test/opencv/yolov3.weights"
names_path = "D:/test/opencv/coco.names"

# --- Load class names ---
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Assign random colors for visualization
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# --- Load the YOLO model ---
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# --- Get output layer names ---
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# --- Start webcam ---
cap = cv2.VideoCapture(0)

# --- IoU function to measure box overlap ---
def box_overlap(box1, box2):
    x1, y1, w1, h1 = box1 #chairs
    x2, y2, w2, h2 = box2 #persons

    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    if x_overlap <= 0 or y_overlap <= 0:
        return 0  # No overlap

    intersection_area = x_overlap * y_overlap
    chair_area = w1 * h1
    person_area = w2 * h2
    iou = intersection_area / (chair_area + person_area - intersection_area)
    return iou

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.blur(frame, (8, 6))
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, class_ids, confidences = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.8 and classes[class_id] in ['person', 'chair']:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    chairs, persons = [], []

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            label = classes[class_ids[i]]
            color = colors[class_ids[i]]

            x, y, w, h = box
            if label == 'chair':
                chairs.append(box)
            elif label == 'person':
                persons.append(box)

            # --- Black box behind label ---
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    occupied_count = 0
    for chair in chairs:
        occupied = False
        for person in persons:
            iou = box_overlap(chair, person)
            if iou > 0.1:
                occupied = True
                break
        x, y, w, h = chair
        status = "Occupied" if occupied else "Vacant"
        status_color = (0, 0, 255) if occupied else (0, 255, 0)

        # --- Black box behind seat status ---
        (text_w, text_h), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y + h + 5), (x + text_w, y + h + 5 + text_h + 5), (0, 0, 0), -1)
        cv2.putText(frame, status, (x, y + h + text_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        occupied_count += int(occupied)

    total_chairs = len(chairs)
    vacant_count = total_chairs - occupied_count
    summary = f"Occupied: {occupied_count} | Vacant: {vacant_count}"

    # --- Black box behind summary text ---
    cv2.putText(frame, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # Black shadow (thicker)
    cv2.putText(frame, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White text
    # Text with black outline
    cv2.putText(frame, f"Total Chairs: {total_chairs}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # Outline (black, thicker)
    cv2.putText(frame, f"Total Chairs: {total_chairs}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Foreground (white)

    cv2.imshow("Seat Occupancy Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Release resources ---
cap.release()
cv2.destroyAllWindows()
