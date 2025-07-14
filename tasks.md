# Entrance Camera System — Specification Document

## Overview
Implement an entrance camera mechanism based on image processing, consisting of three processes running in parallel:
- Two separate **EXE processes** (on Windows) communicating with each other.
- A third process for monitoring and analysis.

---

## Processes

### Process 1 — Image Processing (Client)
Runs on a computer connected to a camera (or reading from video files).

#### 1. Histogram Stretching
- Perform histogram stretching to improve image quality. ✅
  - Which channel(s) should be stretched? (e.g., luminance, grayscale, etc.) ✅
  - Send relevant parameters for the histogram stretch. ✅
    - Which parameters are important? (e.g., min/max intensity, clipping thresholds) ✅
  - Send the runtime of the algorithm to the Data Display Process. ✅

#### 2. Face Detection
- Run a face detection algorithm on the current frame. ✅
  - Send data to the remote computer: 
    - Number of detected faces in the image.✅
    - Location (bounding boxes) of each detected face. ✅
  - Send the runtime of the algorithm to the Data Display Process. ✅

#### 3. Transformation Computation ✅
- Compute a transformation matrix (9 elements) between the current frame and the previous frame.
  - How can the transformation be used to improve the performance of face detection?
    - Improve average runtime. ✅
    - Improve algorithm accuracy. ✅
  - Send the transformation matrix to the Data Display Process. ✅
  - Send the runtime of the algorithm to the Data Display Process.  ✅

#### Sending Data Per Frame ✅
- Current frame number. ✅
- Send a **downscaled image** (by a factor of 4 in each axis) to the remote computer. ✅

---

### Process 2 — Data Display (Server)
This process does not perform calculations — only displays and logs data.

#### Responsibilities
- Receive data from the Image Processing process in real-time:
  - Frame number. ✅
  - Image. ✅
  - Histogram stretch parameters. ✅
  - Face detection data. ✅
  - Transformation data. ✅
- Save the data to files on disk. ✅
- Display the image in real-time. ✅
- Overlay real-time data on the live video: ✅
  - Frame number. ✅
  - Video rate in FPS (frames per second). ✅ 🔱
    - Aim for as high an FPS as possible (see also section 3.1). ✅
  - Display the most recently detected faces — draw bounding boxes around each detected face. ✅

---

### Process 3 — Data Analysis  ✅
This process runs **offline**, after the two previous processes have finished.  
It receives the original video and the saved data files and performs analysis.

#### Outputs
- Generate graphs of algorithm runtimes:
  - Per frame and as an average across all frames:
    - Histogram stretching.
    - Face detection.
    - Transformation computation.
- Validate correctness:
  - Face detection — success rate per frame and average success rate.
    - How can success rate be calculated? (e.g., comparing detected faces against ground truth)
  - Generate a graph of success rate per frame.
  - Transformation computation — accuracy of the calculation (in grayscale intensity units), per frame and on average.
    - How can accuracy be calculated? (e.g., comparing against expected transformation or measuring residual error)
  - Generate a graph of calculation accuracy per frame.

---

## General Requirements
- Implementation in **Python** or **C++**
- Prepare at least **three “interesting” videos**, demonstrating and validating the software:
  1. Video with multiple faces per frame.
  2. Video with a static camera.
  3. Video with motion between frames (non-static camera).
  4. A video combining all of the above.
- Deliverables:
  - 3 Python (or C++) scripts.
  - 3 sample videos.
  - 3 full reports.

---

## Deliverables
✅ 3 executables (processes)  
✅ 3 videos as test cases  
✅ 3 detailed analysis reports  

---

**Good luck!**
