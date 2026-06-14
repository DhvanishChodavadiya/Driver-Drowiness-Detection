**Main Functionalities**
1> Eye Aspect Ratio (EAR)
The heart of the entire system. Calculates how open/closed the eye is using 6 facial landmark points.

2> Auto Calibration
Makes the system adaptive to every person. Instead of hardcoded threshold, it measures each driver's natural eye openness first.

3> Per-Eye Independent Threshold
Handles asymmetric faces. Each eye gets its own threshold so one naturally smaller eye doesn't cause false alerts.

4> Driver Profile System
Saves time for returning drivers. No recalibration needed — just run with your name and detection starts instantly.

5> Drowsiness Alert
The safety output. Triggers audio + visual alert when eyes stay closed too long.

6> Live HUD Display
Real-time feedback so you can see exactly what the system is measuring.

**Current Functionalities**
1> Eye Aspect Ratio (EAR) — measures eye openness using facial landmarks
2> Drowsiness detection — triggers alert after consecutive closed-eye frames
3> Audio alert — plays music.wav when drowsiness detected
4> Visual alert — red ALERT text on screen
5> Eye contour drawing — green/red outline around eyes
6> Per-eye independent EAR — left and right eye measured separately
7> Per-eye independent threshold — each eye gets its own drowsiness threshold
8> Per-eye size measurement — measures actual pixel size of each eye during calibration
9> Alert if either eye drowsy — triggers if left OR right eye closes
10> Auto calibration phase — collects 100 frames of open-eye data on startup
11> Auto threshold calculation — threshold = baseline EAR × 0.75
12> Blink filter — ignores blinks during calibration for clean baseline
13> Progress bar — shows calibration progress on screen
14> Live EAR display — shows real-time EAR values during calibration
15> Safety clamp — keeps threshold between 0.18–0.32 to prevent bad calibration
16> Save driver profile — press [S] to save calibration to JSON
17> Load driver profile — automatically loads saved profile on startup
18> Multiple driver support — each driver saved separately by name
19> Skip calibration — if profile exists, jumps straight to detection
20> Recalibrate anytime — press [R] to redo calibration
21> Live L-EAR display — shows left eye EAR + threshold on screen
22> Live R-EAR display — shows right eye EAR + threshold on screen
23> Driver name display — shows current driver name on screen
24> Eye size display — shows calibrated eye size in pixels
25> Eye color feedback — green contour = open, red contour = closed
26> [S] Save profile — saves current calibration
27> [R] Recalibrate — restarts calibration for new driver or lighting
28> [Q] Quit — exits cleanly

---Muskan & Pankaj---
29> Yawn detection - Detects driver yawn along with eyes
30> Head pose - Detects driver distraction using head pose
31> Mobile detection - Detects driver hands 