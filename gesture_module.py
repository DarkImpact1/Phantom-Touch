import numpy as np
import math
import pyautogui


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

def left_click(hand_landmarks):
    """Triggers a left click when the index and thumb tips are close."""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    # Calculate Euclidean distance
    distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance


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
