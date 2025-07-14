import cv2, numpy as np, json, sys, os
import insightface
from insightface.app import FaceAnalysis

def analyze_video(video_path, output_path='output_offline/trusted_retinaface.json'):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video.")
        return

    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use GPUExecutionProvider if available
    app.prepare(ctx_id=0)

    results_list = []
    frame_id = 0

    print("🔍 Running RetinaFace (InsightFace) offline...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        faces = app.get(frame)
        results_list.append({
            "frame_id": frame_id,
            "face_count": len(faces)
        })

        frame_id += 1

    cap.release()

    with open(output_path, 'w') as f:
        json.dump(results_list, f, indent=2)

    print(f"✅ Saved results for {frame_id} frames to: {output_path}")

if __name__ == "__main__":
    video_path = r"C:\Users\Shahar Golan\VisualStudioProjects\Face_Detaction\input\2.mp4"
    analyze_video(video_path)