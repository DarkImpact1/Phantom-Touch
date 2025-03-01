# PhantomTouch - AI-Powered Gesture-Based Mouse Control

## Overview
PhantomTouch is an AI-powered system that enables touchless control of a laptop mouse using hand gestures. It leverages **MediaPipe Hand Tracking**, **OpenCV**, and **PyAutoGUI** to detect hand movements and perform actions such as:

- **Mouse movement** using the index finger.
- **Left & Right click** detection using pinch gestures.
- **Scrolling** using four-finger movement (index, middle, ring, pinky).
- **Activation & Deactivation** of controls based on finger positions.

## Features
- **Smooth Cursor Control:** Uses exponential smoothing to improve movement precision.
- **Click Actions:** Left-click and right-click detection via pinch gestures.
- **Scroll Feature:** Scrolls up/down based on vertical movement of four fingers.
- **Real-time Hand Tracking:** Powered by **MediaPipe Hands** for accurate gesture recognition.
- **Optimized for Performance:** Efficient tracking with minimal latency.

## Installation
```bash
git clone https://github.com/DarkImpact1/Phantom-Touch.git
cd Phantom-Touch
pip install -r requirements.txt
```

## Usage
Run the program with:
```bash
python main.py
```

## Gesture Controls
| Gesture | Action |
|---------|--------|
| ☝ (Index finger up) | Activate mouse movement |
| ✌ (Index & middle fingers up) | Activate click mode |
| ✋ (All fingers up) | Deactivate all controls |
| 🖖 (Four fingers up, moving up/down) | Scroll up/down |

## File Structure
```
PhantomTouch/
├── gesture_module.py  # Contains gesture detection functions
├── main.py            # Main program for gesture-based mouse control
└── README.md          # Documentation
```

## Code Explanation
### **Mouse Movement (`move_cursor`)**
- Uses index finger's position to move the cursor smoothly.
- Applies **exponential smoothing** to reduce jitter.

### **Click Detection (`detect_left_click_pinch`, `detect_right_click_pinch`)**
- Detects pinch gestures between the thumb and index/middle finger.
- Triggers left or right click based on the detected pinch.

### **Scroll Function (`scroll_screen`)**
- Detects vertical movement of **four fingers (index, middle, ring, pinky)**.
- If fingers move downward, it scrolls down; if they move upward, it scrolls up.
- Uses **threshold-based movement detection** to prevent accidental scrolling.

## Contributions
Feel free to modify and improve the project! Fork, clone, and submit a pull request.

## License
MIT License © 2025 PhantomTouch

