import cv2
import mediapipe as mp
import pyautogui
from gesture_module import fingers_up, Smoother, move_cursor, detect_right_click_pinch, detect_left_click_pinch, scroll_screen

# Control mode states
MODES = {"mouse": "Mouse Movement", "click": "Click Mode", "scroll": "Scroll Mode","none": "All Control Deactivated"}
mode = "none"  # Default mode

def main():
    global mode

    # Initialize MediaPipe Hand module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # Webcam input
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open webcam. Please ensure the webcam is connected and accessible.")
    except Exception as e:
        print(e)
        return

    screen_width, screen_height = pyautogui.size()

    # Initialize Smoother with a larger window size
    smoother = Smoother(window_size=2)

    prev_x, prev_y = 0, 0  # Initialize previous coordinates for smoothing
    alpha = 0.2  # Smoothing factor
    prev_scroll_y = 0  # Previous scroll position

    shutdown_count = 0  # Counter for shutdown prevention
    shutdown_threshold = 20  # Frames before shutting down

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        # Flip image for natural movement & convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = handedness.classification[0].label  # "Right" or "Left"

                if label == "Right":  # Process only right hand
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get finger status
                    finger_states = fingers_up(hand_landmarks)

                    # Set mode based on finger states
                    if finger_states == [1, 0, 0, 0, 0]:
                        mode = "mouse"
                    elif finger_states == [1, 1, 0, 0, 0]:
                        mode = "click"
                    elif finger_states == [1, 1, 1, 1, 0]:
                        mode = "scroll"
                    elif finger_states == [1, 1, 1, 1, 1]:
                        mode = "none"
                    elif finger_states == [0, 1, 0, 0, 0]:  # Shutdown gesture
                        shutdown_count += 1
                        if shutdown_count >= shutdown_threshold:
                            print("Shutting down...")
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                    else:
                        shutdown_count = 0  # Reset shutdown counter if gesture isn't detected continuously

                    # Display mode status
                    cv2.putText(frame, MODES[mode], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Display finger states
                    finger_labels = ["Index", "Middle", "Ring", "Pinky", "Thumb"]
                    for i, state in enumerate(finger_states):
                        cv2.putText(frame, f"{finger_labels[i]}: {'Up' if state else 'Down'}", (50, 100 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Perform actions based on mode
                    if mode == "mouse":
                        cv2.rectangle(frame, (10, 10), (630, 470), (0, 255, 0), 2)  # Green border for mouse mode
                        prev_x, prev_y, smooth_x, smooth_y = move_cursor(hand_landmarks, screen_width, screen_height, alpha, prev_x, prev_y)
                        pyautogui.moveTo(smooth_x, smooth_y, _pause=False)
                    elif mode == "click":
                        cv2.rectangle(frame, (10, 10), (630, 470), (255, 0, 0), 2)  # Blue border for click mode
                        detect_left_click_pinch(frame, hand_landmarks)
                        detect_right_click_pinch(frame, hand_landmarks)
                    elif mode == "scroll":
                        cv2.rectangle(frame, (10, 10), (630, 470), (0, 0, 255), 2)  # Red border for scroll mode
                        prev_scroll_y = scroll_screen(hand_landmarks, screen_height, prev_scroll_y)  # Scroll mode

        # Display output
        cv2.imshow("PhantomTouch - Right Hand Cursor Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()