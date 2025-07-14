import socket, pickle, cv2, numpy as np, time, csv, os

def html_color(c): c=c.lstrip('#'); return (int(c[4:6],16),int(c[2:4],16),int(c[0:2],16))

class VideoUDPServer:
    def __init__(self, host='localhost', port=5555):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.fps, self.prev_time = 0, time.time()
        self.output_dir, self.logfile, self.writer, self.video_name = None, None, None, None
        print(f"🚀 Server running on {host}:{port}")

    def start(self):
        while True:
            packet,_ = self.sock.recvfrom(65536)
            data = pickle.loads(packet)
            # initialize output directory and CSV file if not done yet
            if self.video_name != data['video_name']:
                if self.logfile: self.logfile.close()
                self.video_name = data['video_name']
                self.output_dir = os.path.join("output", self.video_name)
                os.makedirs(self.output_dir, exist_ok=True)
                self.logfile = open(os.path.join(self.output_dir,"session_data.csv"),"w",newline="")
                self.writer = csv.DictWriter(self.logfile, fieldnames=[
                    "frame", "face_count", "faces", "stretch_runtime_ms", "face_runtime_ms", "transform_runtime_ms", "transform_matrix"
                ])
                self.writer.writeheader()
                print(f"💾 Logging to {self.output_dir}/session_data.csv")

            frame = cv2.imdecode(np.frombuffer(data['frame'],np.uint8),cv2.IMREAD_COLOR)
            faces, face_count, H = data['faces'], data['face_count'], np.array(data['transform_matrix'])
            sp, fr, tr = data['histogram_stretch'], data['face_runtime_ms'], data['transform_runtime_ms']

            self.writer.writerow({
                "frame": data['frame_number'], "face_count": face_count, "faces": faces,
                "stretch_runtime_ms": sp.get("runtime_ms",0), "face_runtime_ms":fr, "transform_runtime_ms":tr,
                "transform_matrix":H.tolist()
            })

            now = time.time(); dt = now-self.prev_time
            if dt>0: self.fps = 0.9*self.fps+0.1*(1/dt)
            self.prev_time = now

            for(x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),html_color("#00FF00"),2)

            cv2.putText(frame, f"Frame:{data['frame_number']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, html_color("#FF0062"), 2)
            cv2.putText(frame, f"FPS:{self.fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .7, html_color("#00FFFF"), 2)
            cv2.putText(frame, f"Faces:{face_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, html_color("#C300FF"), 2)
            # Removed the runtime metrics line below
            # cv2.putText(frame, f"Stretch:{sp.get('runtime_ms',0):.1f}ms Face:{fr:.1f}ms Transform:{tr:.1f}ms", (10, 90),
            #             cv2.FONT_HERSHEY_SIMPLEX, .5, html_color("#FFA500"), 2)


            cv2.imshow("Server", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break

        if self.logfile: self.logfile.close()
        self.sock.close(); cv2.destroyAllWindows()
        print("🛑 Server shut down.")

if __name__ == "__main__":
    VideoUDPServer().start()
