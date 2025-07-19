import cv2
import webbrowser
import mediapipe as mp
import time

# Genre playlist map (1–5 fingers)
genre_playlists = {
    1: ("Pop", "https://www.youtube.com/watch?v=ekr2nIex040&list=PLos7xCCYivJ9Oq4pNK9-D1ESAwPkVeveh&shuffle=1"),
    2: ("Hip Hop", "https://www.youtube.com/watch?v=begbi44ZLqw&list=PLDIoUOhQQPlXFSnCfj8HuVhOUSC0QwxYD&shuffle=1"),
    3: ("Rock", "https://www.youtube.com/watch?v=pAgnJDJN4VA&list=RDQM76o5xfhxapw&shuffle=1"),
    4: ("Indie", "https://www.youtube.com/watch?v=b8-tXG8KrWs&list=PLOhV0FrFphUfHqxfhIBju7zu_2CTqG01F&shuffle=1"),
    5: ("Bollywood", "https://www.youtube.com/watch?v=0WtRNGubWGA&list=RDQMQL-ueLciz1Q&shuffle=1")
}

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Open webcam
cap = cv2.VideoCapture(0)

# Cooldown setup
last_opened_time = 0
cooldown = 5  # seconds

# Finger counting function
def count_fingers(hand_landmarks):
    finger_tips_ids = [8, 12, 16, 20]
    fingers = 0

    # Detect hand side (mirror view)
    is_right_hand = hand_landmarks.landmark[5].x > hand_landmarks.landmark[17].x

    # Thumb logic (based on hand side)
    if is_right_hand:
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers += 1
    else:
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers += 1

    # Other fingers
    for tip_id in finger_tips_ids:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers += 1

    return fingers

# Main loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror effect
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)
            fingers_up = count_fingers(hand_landmark)

            current_time = time.time()
            if fingers_up in genre_playlists and (current_time - last_opened_time > cooldown):
                playlist_name, playlist_url = genre_playlists[fingers_up]
                print(f"Opening {playlist_name} playlist for {fingers_up} fingers")
                webbrowser.open(playlist_url)
                last_opened_time = current_time

            # Always show current finger count
            cv2.putText(img, f"Fingers: {fingers_up}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Hand Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()