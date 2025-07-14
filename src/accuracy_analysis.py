import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def get_video_choice():
    """Ask user to choose which video analysis to compare"""
    input_folder = r"C:\Users\Shahar Golan\VisualStudioProjects\Face_Detaction\input"
    
    # Get all video files in input folder
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not video_files:
        print("❌ No video files found in input folder!")
        return None
    
    print("🎬 Available videos for accuracy analysis:")
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        
        # Check if corresponding analysis files exist
        json_path = f"output_offline/{i}/trusted_retinaface_{i}.json"
        csv_path = f"output/{i}/session_data.csv"
        
        status = "✅" if os.path.exists(json_path) and os.path.exists(csv_path) else "❌"
        print(f"{i}. {video_name} {status}")
    
    while True:
        try:
            choice = int(input(f"\nSelect video for analysis (1-{len(video_files)}): "))
            if 1 <= choice <= len(video_files):
                return str(choice)
            else:
                print(f"Please enter a number between 1 and {len(video_files)}")
        except ValueError:
            print("Please enter a valid number")

def load_trusted_data(json_path):
    """Load the trusted RetinaFace results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to dictionary for easy lookup
    trusted_faces = {}
    for entry in data:
        trusted_faces[entry['frame_id']] = entry['face_count']
    
    return trusted_faces

def load_session_data(csv_path):
    """Load the session data from CSV file"""
    df = pd.read_csv(csv_path)
    
    # Create dictionary for easy lookup
    session_faces = {}
    for _, row in df.iterrows():
        session_faces[row['frame']] = row['face_count']
    
    return session_faces

def calculate_accuracy(trusted_data, session_data):
    """Calculate frame-by-frame accuracy using MAE"""
    frames = []
    trusted_counts = []
    session_counts = []
    absolute_errors = []
    
    # Get common frames (intersection of both datasets)
    common_frames = set(trusted_data.keys()) & set(session_data.keys())
    
    for frame in sorted(common_frames):
        trusted_count = trusted_data[frame]
        session_count = session_data[frame]
        
        # Calculate absolute error
        absolute_error = abs(trusted_count - session_count)
        
        frames.append(frame + 1)  # 1-indexed for display
        trusted_counts.append(trusted_count)
        session_counts.append(session_count)
        absolute_errors.append(absolute_error)
    
    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(absolute_errors) if absolute_errors else 0
    
    return frames, trusted_counts, session_counts, mae

def plot_accuracy_analysis(frames, trusted_counts, session_counts, mae, output_path='accuracy_analysis.png'):
    """Create a single face detection comparison plot"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot face counts comparison
    ax.plot(frames, trusted_counts, 'g-', label='Trusted (RetinaFace)', linewidth=2, alpha=0.8)
    ax.plot(frames, session_counts, 'r--', label='Session Data', linewidth=2, alpha=0.8)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Number of Faces Detected')
    ax.set_title('Face Detection Comparison: Trusted vs Session Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to show only integers (0, 1, 2, ...)
    max_faces = max(max(trusted_counts), max(session_counts))
    ax.set_yticks(range(0, max_faces + 1))
    ax.set_ylim(-0.1, max_faces + 0.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_path

def print_accuracy_stats(trusted_counts, session_counts, mae):
    """Print accuracy statistics using MAE"""
    total_frames = len(trusted_counts)
    
    # Calculate match accuracy for debugging
    correct_frames = sum(1 for t, s in zip(trusted_counts, session_counts) if t == s)
    match_accuracy = (correct_frames / total_frames) * 100 if total_frames > 0 else 0
    
    print("=" * 60)
    print("FACE DETECTION ACCURACY ANALYSIS")
    print("=" * 60)
    print(f"Total frames analyzed: {total_frames}")
    print(f"Correctly detected frames: {correct_frames}")
    print(f"Incorrectly detected frames: {total_frames - correct_frames}")
    print(f"\nTOTAL ACCURACY: {match_accuracy:.2f}%")
    print("=" * 60)
    
    # Additional statistics
    trusted_total = sum(trusted_counts)
    session_total = sum(session_counts)
    
    print(f"Total faces (Trusted): {trusted_total}")
    print(f"Total faces (Session): {session_total}")
    print(f"Detection rate: {(session_total / trusted_total * 100):.2f}%" if trusted_total > 0 else "N/A")
    
    # Count distribution
    from collections import Counter
    trusted_dist = Counter(trusted_counts)
    session_dist = Counter(session_counts)
    
    print(f"\nTrusted face count distribution: {dict(trusted_dist)}")
    print(f"Session face count distribution: {dict(session_dist)}")
    
    print(f"\nMean Absolute Error (MAE): {mae:.2f}")
    print(f"Average faces difference: {mae:.2f}")
    print("=" * 60)

def main():
    # Ask user to choose video
    video_number = get_video_choice()
    
    if not video_number:
        print("❌ No video selected. Exiting.")
        return
    
    # Build file paths based on video number
    json_path = f"output_offline/{video_number}/trusted_retinaface_{video_number}.json"
    csv_path = f"output/{video_number}/session_data.csv"
    output_plot = f"face_detection_accuracy_analysis_{video_number}.png"
    
    # Check if files exist
    if not os.path.exists(json_path):
        print(f"❌ Trusted data file not found: {json_path}")
        print("Run the Trusted_Algorithm.py first to generate trusted data.")
        return
    
    if not os.path.exists(csv_path):
        print(f"❌ Session data file not found: {csv_path}")
        print("Run the client/server analysis first to generate session data.")
        return
    
    print(f"🎯 Analyzing video #{video_number}")
    print("Loading data...")
    
    # Load data
    trusted_data = load_trusted_data(json_path)
    session_data = load_session_data(csv_path)
    
    print(f"Loaded {len(trusted_data)} trusted frames and {len(session_data)} session frames")
    
    # Calculate accuracy
    frames, trusted_counts, session_counts, mae = calculate_accuracy(trusted_data, session_data)
    
    # Create plot
    print(f"\nGenerating accuracy plot...")
    plot_path = plot_accuracy_analysis(frames, trusted_counts, session_counts, mae, output_plot)
    print(f"Plot saved to: {plot_path}")
    
    # Print accuracy statistics
    print_accuracy_stats(trusted_counts, session_counts, mae)

if __name__ == "__main__":
    main()
