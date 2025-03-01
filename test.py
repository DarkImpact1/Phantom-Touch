import cv2
import mediapipe as mp
import pyautogui
from gesture_module import fingers_up, Smoother, move_cursor,left_click



if __name__ == "__main__":
    # Initialize MediaPipe Hand module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # Webcam input
    cap = cv2.VideoCapture(0)
    screen_width, screen_height = pyautogui.size()

    # Initialize Smoother with a larger window size
    smoother = Smoother(window_size=10)

    prev_x, prev_y = 0, 0  # Initialize previous coordinates for exponential smoothing
    alpha = 0.2  # Smoothing factor

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip image for natural movement & convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = handedness.classification[0].label  # "Right" or "Left"

                if label == "Right":  # Process only right hand
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get finger status
                    finger_states = fingers_up(hand_landmarks)

                    if finger_states[0] == 1 and finger_states != [1,1,1,1,1]:  # Index finger up, others down
                        print("Mouse movement mode activated")
                        ll = move_cursor(hand_landmarks, screen_width, screen_height, alpha, prev_x, prev_y)
                        smooth_x, smooth_y,prev_x,prev_y = ll[2], ll[3], ll[0], ll[1]
                        # Move cursor with optimized PyAutoGUI
                        pyautogui.moveTo(smooth_x, smooth_y, _pause=False)
                    
                    # Stop movement only when all fingers are open
                    elif finger_states == [0,0, 1, 1, 1]:
                        print("except thumb and index all fingers are up")
                        # distance = left_click(hand_landmarks)
                        # if distance < 0.05:  # Adjust this threshold based on testing
                        #     pyautogui.click()
                        continue  # Do nothing, preventing movement

        # Display output
        cv2.imshow("PhantomTouch - Right Hand Cursor Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

