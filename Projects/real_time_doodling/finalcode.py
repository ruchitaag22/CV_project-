import cv2
import mediapipe as mp
import numpy as np


mpHands = mp.solutions.hands
handTracker = mpHands.Hands()
drawing_utils = mp.solutions.drawing_utils


cam = cv2.VideoCapture(0)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prevX, prevY = 0, 0
drawing_enabled = True  


color_map = {
    'w': (255, 255, 255),
    'r': (0, 0, 255),
    'g': (0, 255, 0),
    'b': (255, 0, 0),
    'y': (0, 255, 255),
    'p': (255, 0, 255),
    'c': (255, 255, 0),
    'o': (0, 165, 255)
}
current_color = color_map['w']

def draw(canvas_img, start, end, color):
    cv2.line(canvas_img, start, end, color, 4)

def erase(canvas_img, center):
    cv2.circle(canvas_img, center, 80, (0, 0, 0), -1)


while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = handTracker.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = hand.landmark

        drawing_utils.draw_landmarks(
            frame, hand, mpHands.HAND_CONNECTIONS,
            drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        tip = landmarks[mpHands.HandLandmark.INDEX_FINGER_TIP]
        mcp = landmarks[mpHands.HandLandmark.INDEX_FINGER_MCP]

        x, y = int(tip.x * 640), int(tip.y * 480)

        
        if tip.y > mcp.y:
            erase(canvas, (x, y))
            prevX, prevY = 0, 0  
        else:
            if drawing_enabled:
                if prevX and prevY:
                    draw(canvas, (prevX, prevY), (x, y), current_color)
                prevX, prevY = x, y
            else:
                prevX, prevY = 0, 0
    else:
        prevX, prevY = 0, 0

    cv2.imshow("Live Feed (with Hand Landmarks)", frame)
    cv2.imshow("Sketch Canvas", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif chr(key) in color_map:
        current_color = color_map[chr(key)]
        print(f"Selected color: {chr(key).upper()}")
    elif key == ord('d'):
        drawing_enabled = not drawing_enabled
        print(f"Drawing {'enabled' if drawing_enabled else 'disabled'}")

cam.release()
cv2.destroyAllWindows()
