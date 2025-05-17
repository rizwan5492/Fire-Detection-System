# Fire Detection System

A Python-based fire detection system using computer vision techniques with OpenCV, featuring real-time processing, audio alerts, and a GUI for HSV threshold adjustment. This project was developed as a learning exercise in software engineering, combining image processing, automation, and safety applications.

## Features
- Detects fire-like colors using HSV color space.
- Real-time video processing from a webcam.
- Audio alert (using an alarm sound) when fire is detected.
- GUI to dynamically adjust HSV thresholds for accurate detection.
- Logging of detection events to a file.
- Motion detection to reduce false positives.
- Visual feedback with bounding boxes and text overlays.

## Prerequisites
- **Python 3.10+**
- Required libraries:
  - `opencv-python`
  - `numpy`
  - `pygame`
- An audio file (`alarm.wav`) for alerts.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Fire-Detection-System.git
   cd Fire-Detection-System
