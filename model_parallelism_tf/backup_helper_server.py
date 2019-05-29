import os
import sys
import tensorflow as tf

import backup_train_1

IP_ADDRESS = backup_train_1.IP_ADDRESS
START_PORT = backup_train_1.START_PORT

n_ports = int(sys.argv[1])
PORTS = [str(START_PORT + i) for i in range(n_ports)]
task_idx = int(sys.argv[-1])

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[-1]

workers = [IP_ADDRESS + ":" + PORT for PORT in PORTS]
cluster_spec = tf.train.ClusterSpec({'worker': workers})

server = tf.train.Server(cluster_spec, job_name='worker', task_index=task_idx)

server.join()

