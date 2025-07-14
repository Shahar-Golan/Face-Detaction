import subprocess, threading, time, os, sys

INPUT_DIR = 'input'

def reader_thread(pipe, name):
    for line in iter(pipe.readline, ''):
        if line: print(f"[{name}] {line.strip()}")
    pipe.close()

def start_process(cmd, name):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    t = threading.Thread(target=reader_thread, args=(proc.stdout, name), daemon=True)
    t.start()
    return proc, t

def wait_for_process(proc, name):
    try: proc.wait(); print(f"[{name}] exited with code {proc.returncode}")
    except KeyboardInterrupt:
        print(f"[{name}] interrupted."); proc.terminate()
        try: proc.wait(timeout=3)
        except subprocess.TimeoutExpired: proc.kill()
        print(f"[{name}] killed.")

def main():
    server_script, client_script = os.path.abspath("server.py"), os.path.abspath("client.py")
    print("🚀 Starting server...")
    server_proc, server_thread = start_process([sys.executable, server_script], "SERVER")
    time.sleep(1)

    videos = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.mp4','.avi','.mov'))])
    if not videos:
        print("❌ No videos found in input/"); server_proc.terminate(); server_proc.wait(); return

    for video in videos:
        video_path = os.path.join(INPUT_DIR, video)
        print(f"\n🎥 Processing: {video}")
        client_proc, client_thread = start_process([sys.executable, client_script, video_path], "CLIENT")
        wait_for_process(client_proc, "CLIENT")
        print(f"✅ Finished: {video}")
        time.sleep(1)

    print("\n🎯 All videos processed. Shutting down server...")
    server_proc.terminate()
    try: server_proc.wait(timeout=3)
    except subprocess.TimeoutExpired: server_proc.kill()
    print("🛑 Server shut down.")

if __name__ == "__main__":
    main()
