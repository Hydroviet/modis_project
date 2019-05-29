import subprocess
import tensorflow as tf
from utils import get_available_gpus

GPUS = 4
GPUS = min(GPUS, len(get_available_gpus()))
python_cmd = "python"


def run_server(idx):
    subprocess.call([python_cmd, "helper_server.py", str(GPUS), str(idx)])

processes = []
for i in range(1, GPUS):
    p = Process(target=run_server, args=(i,))
    processes.append(p)

for p in processes:
    p.start()

for p in processes:
    p.join()
