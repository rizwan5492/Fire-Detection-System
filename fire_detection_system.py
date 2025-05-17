import cv2
import numpy as np
import pygame
import tkinter as tk
from tkinter import ttk
import time
from datetime import datetime
import os

# Initialize pygame mixer for audio
pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound("alarm.wav")  # Load WAV file
except FileNotFoundError:
    print("Error: alarm.wav not found. Please ensure alarm.wav is in the project directory.")
    exit()

# Global variables for HSV thresholds
hsv_lower = [0, 100, 100]  # Tightened for bright orange-red fire
hsv_upper = [20, 255, 255]  # Adjust via GUI
fire_detected = False
last_detection_time = 0
log_file = "fire_detection_log.txt"
prev_frame = None  # For motion detection

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Frame skip counter
frame_skip = 2  # Process every 2nd frame
frame_count = 0

# Create GUI for HSV adjustment
class HSVAdjuster:
    def __init__(self, root):
        self.root = root
        self.root.title("HSV Threshold Adjuster")
        
        # Labels and sliders for lower HSV
        tk.Label(root, text="Lower HSV").grid(row=0, column=0, columnspan=2)
        tk.Label(root, text="Hue (0-180):").grid(row=1, column=0)
        self.lower_h = tk.Scale(root, from_=0, to=180, orient=tk.HORIZONTAL, command=self.update_hsv)
        self.lower_h.set(hsv_lower[0])
        self.lower_h.grid(row=1, column=1)
        
        tk.Label(root, text="Saturation (0-255):").grid(row=2, column=0)
        self.lower_s = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_hsv)
        self.lower_s.set(hsv_lower[1])
        self.lower_s.grid(row=2, column=1)
        
        tk.Label(root, text="Value (0-255):").grid(row=3, column=0)
        self.lower_v = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_hsv)
        self.lower_v.set(hsv_lower[2])
        self.lower_v.grid(row=3, column=1)
        
        # Labels and sliders for upper HSV
        tk.Label(root, text="Upper HSV").grid(row=4, column=0, columnspan=2)
        tk.Label(root, text="Hue (0-180):").grid(row=5, column=0)
        self.upper_h = tk.Scale(root, from_=0, to=180, orient=tk.HORIZONTAL, command=self.update_hsv)
        self.upper_h.set(hsv_upper[0])
        self.upper_h.grid(row=5, column=1)
        
        tk.Label(root, text="Saturation (0-255):").grid(row=6, column=0)
        self.upper_s = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_hsv)
        self.upper_s.set(hsv_upper[1])
        self.upper_s.grid(row=6, column=1)
        
        tk.Label(root, text="Value (0-255):").grid(row=7, column=0)
        self.upper_v = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_hsv)
        self.upper_v.set(hsv_upper[2])
        self.upper_v.grid(row=7, column=1)
        
        # Start button
        tk.Button(root, text="Start Detection", command=self.start_detection).grid(row=8, column=0, columnspan=2)
        
    def update_hsv(self, _=None):
        global hsv_lower, hsv_upper
        hsv_lower = [self.lower_h.get(), self.lower_s.get(), self.lower_v.get()]
        hsv_upper = [self.upper_h.get(), self.upper_s.get(), self.upper_v.get()]
        print(f"Updated HSV: Lower={hsv_lower}, Upper={hsv_upper}")
        
    def start_detection(self):
        self.root.quit()  # Close GUI to start detection

# Log detection events
def log_detection(status):
    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {status}\n")

# Main detection loop
def detect_fire():
    global fire_detected, last_detection_time, frame_count, prev_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
            
        frame_count += 1
        if frame_count % frame_skip != 0:  # Skip frames
            continue
            
        # Convert frame to HSV and grayscale for motion
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Motion detection
        if prev_frame is not None:
            frame_diff = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]
            motion_area = cv2.countNonZero(thresh)
        else:
            motion_area = 0
        prev_frame = gray.copy()
        
        # Create mask for fire colors
        lower_bound = np.array(hsv_lower)
        upper_bound = np.array(hsv_upper)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_area = 0
        fire_bbox = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area for consideration
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.5 <= aspect_ratio <= 2.0 and area > 1000:  # Filter for fire-like shapes
                    fire_area += area
                    if fire_bbox is None:
                        fire_bbox = (x, y, x + w, y + h)
                    else:
                        fire_bbox = (
                            min(fire_bbox[0], x),
                            min(fire_bbox[1], y),
                            max(fire_bbox[2], x + w),
                            max(fire_bbox[3], y + h)
                        )
        
        # Determine fire detection status
        current_time = time.time()
        if fire_area > 1000 and motion_area > 500:  # Combine color and motion
            if not fire_detected:
                print("Fire detected!")
                log_detection("Fire detected")
                fire_detected = True
                last_detection_time = current_time
                if not pygame.mixer.get_busy():
                    alarm_sound.play(loops=-1)  # Play WAV in loop
            # Draw visual alerts
            cv2.putText(frame, "FIRE DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if fire_bbox:
                cv2.rectangle(frame, (fire_bbox[0], fire_bbox[1]), (fire_bbox[2], fire_bbox[3]), (0, 0, 255), 2)
        else:
            if fire_detected and (current_time - last_detection_time > 5):  # 5-second debounce
                print("Fire no longer detected.")
                log_detection("Fire no longer detected")
                fire_detected = False
                alarm_sound.stop()  # Stop WAV
            cv2.putText(frame, "No Fire", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frames
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    # Create and run GUI
    root = tk.Tk()
    app = HSVAdjuster(root)
    root.mainloop()
    
    # Start detection
    detect_fire()