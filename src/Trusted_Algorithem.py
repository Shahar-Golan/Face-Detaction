import cv2, numpy as np, json, sys, os, glob
import insightface
from insightface.app import FaceAnalysis

def get_video_choice():
    """Ask user to choose which video to analyze"""
    input_folder = r"C:\Users\Shahar Golan\VisualStudioProjects\Face_Detaction\input"
    
    # Get all video files in input folder
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not video_files:
        print("❌ No video files found in input folder!")
        return None, None
    
    print("🎬 Available videos:")
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print(f"{i}. {video_name}")
    
    while True:
        try:
            choice = int(input(f"\nSelect video (1-{len(video_files)}): "))
            if 1 <= choice <= len(video_files):
                selected_video = video_files[choice - 1]
                return selected_video, str(choice)
            else:
                print(f"Please enter a number between 1 and {len(video_files)}")
        except ValueError:
            print("Please enter a valid number")

def analyze_video(video_path, video_number):
    """Analyze video and save results to numbered directory"""
    # Create output directory based on video number
    output_dir = f"output_offline/{video_number}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"trusted_retinaface_{video_number}.json")
    output_path = os.path.join(output_dir, f"trusted_retinaface_{video_number}.json")
    
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
    # Ask user to choose video
    video_path, video_number = get_video_choice()
    
    if video_path and video_number:
        print(f"🎯 Selected: {os.path.basename(video_path)}")
        analyze_video(video_path, video_number)
    else:
        print("❌ No video selected. Exiting.")