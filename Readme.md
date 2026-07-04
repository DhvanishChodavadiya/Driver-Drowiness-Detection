
# Driver Drowsiness Detection and Driver Monitoring System

An AI-powered Driver Monitoring System that detects drowsiness, fatigue, yawning, driver distraction, and mobile phone usage in real time using a camera.

## Features

### Eye Aspect Ratio (EAR)
- Real-time eye openness monitoring
- Independent left and right eye analysis
- Adaptive thresholds for each driver
- Personalized calibration

### PERCLOS Monitoring
- 120-second rolling window
- Measures percentage of eye closure over time
- Driver state classification:
  - ALERT
  - FATIGUE
  - DROWSY
  - HIGH RISK

### Drowsiness Detection
- Detects prolonged eye closure
- Visual and audio alerts
- Personalized thresholds based on calibration

### Yawn Detection
- Mouth Aspect Ratio (MAR) calculation
- Real-time yawn monitoring
- False-positive reduction using cooldown logic

### Driver Distraction Detection
- Head pose estimation
- Detects looking away from the road
- Real-time warning system

### Mobile Phone Detection
- MediaPipe Hand Tracking
- YOLOv8 Phone Detection
- Phone usage timer
- Alert for prolonged phone usage while driving

### Low-Light Monitoring
- Brightness measurement
- Gamma correction
- CLAHE enhancement
- Lighting status indicator

### Driver Profile System
- Saves individual driver calibration
- Personalized EAR thresholds
- Automatic profile loading
- JSON-based profile storage


## Technologies Used

- Python
- OpenCV
- MediaPipe Face Landmarker
- MediaPipe Hand Landmarker
- YOLOv8
- NumPy
- SciPy
- Pygame


## Project Workflow

Camera Input
     ↓
Face Detection
     ↓
EAR Calculation
     ↓
PERCLOS Analysis
     ↓
Yawn Detection
     ↓
Head Pose Estimation
     ↓
Phone Detection
     ↓
Driver State Assessment
     ↓
Audio & Visual Alerts


Installation

pip install opencv-python mediapipe pygame scipy ultralytics numpy

Run

Existing Driver: python Newfunctionalities.py Abhay

New Driver: python Newfunctionalities.py NewDriver

Controls
Key	         Function
S	         Save Driver Profile
R	         Recalibrate
Q	         Quit

Current Functionalities

-Adaptive EAR Thresholds
-Driver Profiles
-Automatic Calibration
-PERCLOS Monitoring
-Drowsiness Detection
-Yawn Detection
-Driver Distraction Detection
-Mobile Phone Detection
-Low-Light Monitoring
-Audio Alerts
-Visual Alerts
