import subprocess
import re
import threading
import time

def bg_iperf_server():
    print("Starting server on port 5202...")
    cmd = ["mx", "h2", "iperf3", "-s", "-i", "1", "-p", "5202"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    pattern = re.compile(r'([\d\.]+)\s+ms\s+\d+/\d+\s+\(([\d\.]+)%\)')
    
    while True:
        line = proc.stdout.readline()
        if not line: break
        print(f"DEBUG Server Line: {line.strip()}")
        match = pattern.search(line)
        if match:
            print(f"MATCHED: Jitter={match.group(1)}, Loss={match.group(2)}")

def bg_iperf_client():
    time.sleep(2)
    print("Starting client on port 5202...")
    cmd = ["mx", "h1", "iperf3", "-c", "10.0.2.2", "-u", "-b", "1M", "-t", "5", "-i", "1", "-p", "5202"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Client finished.")

t1 = threading.Thread(target=bg_iperf_server, daemon=True)
t1.start()
bg_iperf_client()
time.sleep(2)
