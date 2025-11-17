import subprocess
import os
import sys
from datetime import datetime
'''
def run_experiment(script_name):
    """Chạy từng script con và stream output trực tiếp ra terminal"""
    print("=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ Running {script_name} ...")
    print("=" * 80)
   
    process = subprocess.Popen(
        ["python", script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream output real-time
    while True:
        output = process.stdout.readline()
        error = process.stderr.readline()

        if output:
            sys.stdout.write(output)
            sys.stdout.flush()
        if error:
            sys.stderr.write(error)
            sys.stderr.flush()

        if output == '' and error == '' and process.poll() is not None:
            break

    print(f"Finished {script_name}\n")
'''

def run_experiment(script_name):
    print("=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ Running {script_name} ...")
    print("=" * 80)

    process = subprocess.Popen(
        ["python", script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # Hợp nhất log
        universal_newlines=True,
        bufsize=1
    )

    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

    process.wait()
    print(f"Finished {script_name}\n")

if __name__ == "__main__":
    # Chuyển đến thư mục src
    SRC_DIR = os.path.join(os.getcwd(), "src")
    if not os.path.exists(SRC_DIR):
        print(f"[Error] Thư mục src không tồn tại tại: {SRC_DIR}")
        sys.exit(1)
    os.chdir(SRC_DIR)

    scripts = [
        "preprocessing.py",               
        "CNN_train.py",             
        "GRU_train.py",             
        "LSTM_train.py",            
        "XGBoost_train.py",
        #"PhoBERT_train.py"          
    ]

    print(f"\n Starting full pipeline ({len(scripts)} stages)...\n")

    # Lặp qua từng script và chạy lần lượt
    for script in scripts:
        if not os.path.exists(script):
            print(f"[Warning] File {script} không tồn tại trong src/. Bỏ qua.")
            continue
        run_experiment(script)

    print("\nAll experiments completed successfully!\n")
    print("Mở giao diện MLflow bằng lệnh:")
    print("mlflow ui --backend-store-uri file:///D:/Project/mlruns")
    print("Sau đó truy cập: http://localhost:5000 để xem kết quả.")
