import cv2
import mediapipe as mp
import pyautogui
from gesture_module import fingers_up, Smoother, move_cursor, detect_left_click_pinch, detect_right_click_pinch, scroll_screen

# Flags to control actions
activate_mouse = False
activate_click_action = False
scroll_active = False

def main():
    global activate_mouse, activate_click_action, scroll_active

    # Initialize MediaPipe Hand module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # Webcam input
    cap = cv2.VideoCapture(0)
    screen_width, screen_height = pyautogui.size()

    # Initialize Smoother with a larger window size
    smoother = Smoother(window_size=10)

    prev_x, prev_y = 0, 0  # Initialize previous coordinates for smoothing
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

                    # Set modes based on finger states
                    if finger_states == [1, 0, 0, 0, 0]:  # Mouse movement mode
                        activate_mouse = True
                        activate_click_action = False
                        scroll_active = False
                        text = "Mouse Movement Activated"
                    elif finger_states == [1, 1, 0, 0, 0]:  # Click mode
                        activate_click_action = True
                        activate_mouse = False
                        scroll_active = False
                        text = "Click Mode Activated"
                    elif finger_states == [1, 1, 1, 1, 0]:  # Scroll mode
                        scroll_active = True    
                        activate_mouse = False
                        activate_click_action = False
                        text = "Scroll Mode Activated"
                    elif finger_states == [1, 1, 1, 1, 1]:  # Deactivate all modes
                        activate_mouse = False
                        activate_click_action = False
                        scroll_active = False
                        text = "All Control Deactivated"
                    elif finger_states == [0, 1, 1, 1, 0]:  # shutdown the program
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                    else:
                        text = ""

                    # Display status text
                    if text:
                        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Move mouse if active
                    if activate_mouse:
                        ll = move_cursor(hand_landmarks, screen_width, screen_height, alpha, prev_x, prev_y)
                        smooth_x, smooth_y, prev_x, prev_y = ll[2], ll[3], ll[0], ll[1]
                        pyautogui.moveTo(smooth_x, smooth_y, _pause=False)

                    # Detect clicks if active
                    elif activate_click_action:
                        detect_left_click_pinch(frame, hand_landmarks)
                        detect_right_click_pinch(frame, hand_landmarks)
                    
                    elif scroll_active:
                        prev_y = scroll_screen(hand_landmarks, prev_y, screen_height, sensitivity=2)

        # Display output
        cv2.imshow("PhantomTouch - Right Hand Cursor Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
