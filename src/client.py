import socket, cv2, time, pickle, numpy as np, mediapipe as mp, os, sys

class VideoUDPClient:
    def __init__(self, video_path, server_ip='localhost', server_port=5555):
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        # server address
        self.server_addr = (server_ip, server_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # capture object to read frames from the video
        self.cap = cv2.VideoCapture(video_path)
        # Gets a reference to the MediaPipe Face Detection module
        self.mp_face_detection = mp.solutions.face_detection
        # parameters for face detection (adjust as needed)
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.2)

        self.prev_gray, self.prev_faces, self.frame_count = None, [], 0
        # set the interval for face detection (every 10 frames)
        self.detect_interval = 10
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25  # fallback
        self.target_frame_time = 1.0 / fps

    def detect_faces(self, frame):
        faces = []
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        if results.detections:
            for face in results.detections:
                bounding_box = face.location_data.relative_bounding_box
                # convert relative coordinates to absolute pixel values
                im_height, im_width, _ = frame.shape
                x,y,w,h = max(int(bounding_box.xmin*im_width),0), max(int(bounding_box.ymin*im_height),0),int(bounding_box.width*im_width), int(bounding_box.height*im_height)
                # adds the face bounding box to the list.
                faces.append([x,y,w,h])
        return faces

    def compute_transform_lk(self, prev_gray, curr_gray):
        # detect up to 100 good points to track
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        if p0 is None: return np.eye(3)
        # track the movement of the points from perv to curr
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
        good_p0, good_p1 = p0[st==1], p1[st==1]
        if len(good_p0) < 4: return np.eye(3)
        # estimate the affine transformation matrix using RANSAC
        H, _ = cv2.estimateAffinePartial2D(good_p0, good_p1, method=cv2.RANSAC)
        if H is None: return np.eye(3)
        # ensure the transformation matrix is 3x3
        return np.vstack([H, [0,0,1]])

    # applies H to the bounding box roi
    def warp_roi(self, roi, H):
        x,y,w,h = roi
        # construct corners of the bounding box in homogeneous coordinates
        corners = np.array([
            [x,     y,     1],
            [x + w, y,     1],
            [x,     y + h, 1],
            [x + w, y + h, 1]
        ]).T
        warped = H @ corners
        # convert back to pixel coordinates
        warped /= warped[2,:]
        # find the new bounding box coordinates
        x_new,y_new,w_new,h_new = warped[0].min(), warped[1].min(), warped[0].max()-warped[0].min(), warped[1].max()-warped[1].min()
        # return the new bounding box
        return [int(x_new), int(y_new), int(w_new), int(h_new)]

    def histogram_stretch(self, gray):
        t0 = time.time()
        # find the min and max pixel values
        imin, imax = np.min(gray), np.max(gray)
        stretched = gray.copy()
        if imax-imin<1:
            stretched = gray.copy()
        else:
            # stretch the image to cover the full range [0, 255]
            stretched = ((gray-imin)*255/(imax-imin)).clip(0,255).astype(np.uint8)
        return stretched, {'input_min':int(imin),'input_max':int(imax),'output_min':0,'output_max':255,'runtime_ms':round((time.time()-t0)*1000,2)}

    def start(self):
        if not self.cap.isOpened():
            print("❌ Cannot open video."); return

        while True:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret: break

            h,w = frame.shape[:2]
            # downscale the frame by 4
            frame_resized = cv2.resize(frame, (w//4,h//4))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            # A stretched image and dict with stats
            gray_stretched, stretch_params = self.histogram_stretch(gray)

            if self.prev_gray is not None:
                t0 = time.time()
                # compute transformation between prev and curr
                H = self.compute_transform_lk(self.prev_gray, gray_stretched)
                transform_runtime = (time.time()-t0)*1000
            else:
                H, transform_runtime = np.eye(3), 0.0

            # every 10 frames OR if no previous faces detected:
            if self.frame_count%self.detect_interval==0 or not self.prev_faces:
                t0 = time.time()
                faces_detected = self.detect_faces(frame_resized)
                face_runtime = (time.time()-t0)*1000
            # keep previous faces if no new ones detected and save time
            else:
                faces_detected, face_runtime = [], 0.0

            # if we are not in 10 frame interval, use previous faces
            # if no new faces detected, use previous faces
            faces_to_send = [self.warp_roi(r,H) for r in self.prev_faces]
            if faces_detected: faces_to_send = faces_detected

            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY,60])
            data = {
                'video_name': self.video_name, 'frame_number': self.frame_count,
                'frame': buffer.tobytes(), 'faces': faces_to_send, 'face_count': len(faces_to_send),
                'transform_matrix': H.tolist(), 'histogram_stretch': stretch_params,
                'face_runtime_ms': round(face_runtime,2), 'transform_runtime_ms': round(transform_runtime,2)
            }
            # serialize the data and send it over UDP
            packet = pickle.dumps(data)
            if len(packet)<60000: self.sock.sendto(packet, self.server_addr)

            self.prev_gray, self.prev_faces = gray_stretched.copy(), faces_detected or self.prev_faces
            self.frame_count += 1

            elapsed = time.time()-start_time
            if elapsed<self.target_frame_time:
                time.sleep(self.target_frame_time-elapsed)

            if cv2.waitKey(1)&0xFF==ord('q'): break

        self.cap.release(); self.sock.close(); cv2.destroyAllWindows()
        print("✅ Client finished.")

if __name__ == "__main__":
    if len(sys.argv)<2: print("Usage: python client.py input/<video.mp4>")
    else: VideoUDPClient(sys.argv[1]).start()
