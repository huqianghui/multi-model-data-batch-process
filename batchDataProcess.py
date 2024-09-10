import os
import subprocess
from concurrent.futures import ProcessPoolExecutor


def call_process_data_file(file_path):
    subprocess.run(["python3", "dataProcess.py", file_path])

def process_multiple_files(directory, max_workers):
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_process_data_file, file_path) for file_path in file_paths]
        
        for future in futures:
            future.result()  # 等待所有任务完成

# Example usage
if __name__ == "__main__":
    directory = os.getenv("temp_dir")
    process_multiple_files(directory, max_workers=os.cpu_count())