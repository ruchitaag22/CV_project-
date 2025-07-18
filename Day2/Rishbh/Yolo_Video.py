import cv2
import numpy as np

# --- Paths to your YOLOv3 files ---
config_path = "yolov3.cfg"
weights_path = "yolov3.weights"
names_path = "coco.names"

# --- Load class names ---
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate unique colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# --- Load the YOLO model ---
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# --- Start webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Create blob from frame
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass
    outs = net.forward(output_layers)

    # Initialize lists
    class_ids = []
    confidences = []
    boxes = []

    # Process detections
    for output in outs:
        for detection in output:
            scores = detection[5:]  # class probabilities
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes
    if len(indices) > 0:
        for i in indices:
            i = int(i)  # Handle tuple/list format from OpenCV
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the output frame
    cv2.imshow("YOLOv3 Object Detection - Live", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Release resources ---
cap.release()
cv2.destroyAllWindows()
