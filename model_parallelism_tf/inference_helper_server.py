import os
import sys
import tensorflow as tf

import inference

IP_ADDRESS = inference.IP_ADDRESS
START_PORT = inference.START_PORT

n_ports = int(sys.argv[1])
PORTS = [str(START_PORT + i) for i in range(n_ports)]
task_idx = int(sys.argv[-1])

os.environ["CUDA_VISIBLE_DEVICES"] = str(inference.GPU_ID + 1)

workers = [IP_ADDRESS + ":" + PORT for PORT in PORTS]
cluster_spec = tf.train.ClusterSpec({'worker': workers})

server = tf.train.Server(cluster_spec, job_name='worker', task_index=task_idx)

server.join()

