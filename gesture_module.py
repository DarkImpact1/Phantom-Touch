import numpy as np
import math, time
import pyautogui
import cv2

class Smoother:
    """Applies moving average smoothing to cursor movement"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.x_history = []
        self.y_history = []

    def smooth(self, x, y):
        """Applies moving average to smooth cursor movement"""
        self.x_history.append(x)
        self.y_history.append(y)

        if len(self.x_history) > self.window_size:
            self.x_history.pop(0)
            self.y_history.pop(0)

        avg_x = int(np.mean(self.x_history))
        avg_y = int(np.mean(self.y_history))

        return avg_x, avg_y

def fingers_up(hand_landmarks):
    """Check which fingers are up"""
    fingertips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    thumb_tip = 4

    fingers = []

    # Check if fingertips are above their lower joints (considered 'up')
    for tip in fingertips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)  # Finger is up
        else:
            fingers.append(0)  # Finger is down

    # Thumb logic: Left/right comparison as thumbs move sideways
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 2].x:
        fingers.append(1)  # Thumb is up
    else:
        fingers.append(0)  # Thumb is down

    return fingers  # [Index, Middle, Ring, Pinky, Thumb] (1 = Up, 0 = Down)


def move_cursor(hand_landmarks, screen_width, screen_height, alpha, prev_x, prev_y):
    """Move cursor based on index finger position"""
    index_finger = hand_landmarks.landmark[8]  # Index fingertip

    x = int(index_finger.x * screen_width)
    y = int(index_finger.y * screen_height)

    # Apply exponential smoothing
    smooth_x = alpha * x + (1 - alpha) * prev_x
    smooth_y = alpha * y + (1 - alpha) * prev_y
    prev_x, prev_y = smooth_x, smooth_y
    return prev_x, prev_y, smooth_x, smooth_y


click_active = False  # Flag to prevent multiple clicks
last_click_time = 0   # Timestamp to avoid accidental rapid clicks
cooldown_time = 0.3   # Cooldown time (in seconds) between clicks

def detect_left_click_pinch(frame,hand_landmarks, threshold=0.05):
    """Detects a pinch gesture and performs a left-click when detected."""
    global click_active, last_click_time

    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    # Calculate Euclidean distance between thumb tip and index finger tip
    distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    cv2.putText(frame, str(distance), (60, 60), cv2.FONT_HERSHEY_SIMPLEX,0.1 ,(0, 255, 0), 2)
    current_time = time.time()

    if distance < threshold and not click_active:
        if current_time - last_click_time > cooldown_time:  # Check cooldown
            print("Left Click Detected", distance)
            pyautogui.click()
            last_click_time = current_time  # Update last click time
        click_active = True  # Prevent continuous clicking

    elif distance > threshold:
        click_active = False  # Reset when fingers move apart

def detect_right_click_pinch(frame,hand_landmarks, threshold=0.05):
    """
    Detects pinch between thumb and middle finger and performs a right-click.
    """
    thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
    middle_tip = hand_landmarks.landmark[12]  # Middle finger tip

    # Calculate Euclidean distance
    distance = ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5
    cv2.putText(frame, str(distance), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if distance < threshold:
        print("Right Click Detected", distance)
        pyautogui.rightClick()

import pyautogui

def scroll_screen(hand_landmarks, prev_finger_y, screen_height, sensitivity=5):
    """
    Scrolls the screen based on the movement of four fingers (index, middle, ring, pinky).

    Parameters:
        hand_landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): Hand landmarks detected by MediaPipe.
        prev_finger_y (float): Previous average y-coordinate of the four fingertips.
        screen_height (int): Screen height to normalize movement.
        sensitivity (int): Controls scroll speed (higher = faster scrolling).

    Returns:
        float: Updated previous y-coordinate for tracking.
    """
    # Get y-coordinates of four fingertips (index, middle, ring, pinky)
    finger_tips = [8, 12, 16, 20]
    current_finger_y = sum([hand_landmarks.landmark[i].y for i in finger_tips]) / 4  # Average y-position

    # Convert to screen coordinates
    y_position = int(current_finger_y * screen_height)

    # Calculate vertical movement
    movement = y_position - prev_finger_y

    # Scroll based on movement direction
    if abs(movement) > 10:  # Prevent small accidental movements
        scroll_amount = -int(movement / sensitivity)  
        pyautogui.scroll(scroll_amount)

    return y_position  # Update previous y-coordinate
