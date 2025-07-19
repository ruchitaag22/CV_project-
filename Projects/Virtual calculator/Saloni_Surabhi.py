import cv2
import numpy as np
import mediapipe as mp
import time


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils


buttons = [
    ["7", "8", "9", "/"],
    ["4", "5", "6", "*"],
    ["1", "2", "3", "-"],
    ["0", "C", "=", "+"]
]

class Button:
    def __init__(self, pos, text):
        self.pos = pos
        self.text = text
        self.size = (60, 60)

    def draw(self, img):
        x, y = self.pos
        cv2.rectangle(img, (x, y), (x + self.size[0], y + self.size[1]), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, self.text, (x + 20, y + 55), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    def is_hover(self, x, y):
        bx, by = self.pos
        bw, bh = self.size
        return bx < x < bx + bw and by < y < by + bh


button_list = []
for i in range(4):
    for j in range(4):
        button_list.append(Button((j * 70 + 20, i * 70 + 100), buttons[i][j]))


def is_fist(lmList):
    tips = [8, 12, 16, 20] 
    closed = 0
    for tip in tips:
        if lmList[tip][2] > lmList[tip - 2][2]:  
            closed += 1
    return closed >= 3

expression = ""
last_selection_time = 0
selection_delay = 1

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    lmList = []

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        for id, lm in enumerate(handLms.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append((id, cx, cy))

        mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        if lmList and is_fist(lmList):
            
            palm_x, palm_y = lmList[8][1], lmList[8][2]
            cv2.circle(img, (palm_x, palm_y), 10, (0, 255, 0), cv2.FILLED)

            if time.time() - last_selection_time > selection_delay:
                for button in button_list:
                    if button.is_hover(palm_x, palm_y):
                        selected = button.text
                        if selected == "C":
                            expression = ""
                        elif selected == "=":
                            try:
                                expression = str(eval(expression))
                            except:
                                expression = "Error"
                        else:
                            expression += selected
                        last_selection_time = time.time()
                        break  


    for button in button_list:
        button.draw(img)

    
    cv2.rectangle(img, (20, 20), (380, 80), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, expression, (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    #
    cv2.imshow("Gesture Calculator (Fist Only)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
