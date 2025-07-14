import sys, os, pandas as pd, matplotlib.pyplot as plt, numpy as np

def pick_session():
    out = "output"
    sessions = [d for d in os.listdir(out) if os.path.isdir(os.path.join(out,d))]
    if not sessions:
        print("❌ No sessions found in output/")
        sys.exit(1)
    print("Available sessions:")
    for idx, s in enumerate(sessions):
        print(f"{idx+1}. {s}")
    choice = input("Select session number: ")
    try:
        i = int(choice)-1
        return sessions[i]
    except:
        print("❌ Invalid choice"); sys.exit(1)

if len(sys.argv) >= 2:
    video_name = sys.argv[1]
else:
    video_name = pick_session()

path = os.path.join("output", video_name, "session_data.csv")
if not os.path.exists(path): print(f"❌ {path} not found"); sys.exit(1)

df = pd.read_csv(path)
out_dir = os.path.join("output", video_name)

print(f"📄 Loaded {len(df)} frames from {path}")

plt.figure()
plt.plot(df['frame'],df['stretch_runtime_ms'],label='Stretch')
plt.plot(df['frame'],df['face_runtime_ms'],label='Face')
plt.plot(df['frame'],df['transform_runtime_ms'],label='Transform')
plt.xlabel("Frame"); plt.ylabel("ms"); plt.title("Runtimes"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(out_dir,"runtimes_per_frame.png")); plt.close()
print("✅ runtimes_per_frame.png")

print("\nAverage runtimes (ms):")
print(df[['stretch_runtime_ms','face_runtime_ms','transform_runtime_ms']].mean())

print("\n🎉 Analysis done. Outputs in", out_dir)
