#!/bin/bash

# go inside the docker image
nvidia-docker run -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /home/:/home/ -v /tmp/:/tmp/ nvcr.io/nvidia/tensorflow:18.10-py3

cd /home/tftrt/

python3 tftrt_sample_chest.py --native --FP32 --FP16 --INT8 --num_loops 10 --topN 5 --batch_size 1 --log_file log.txt --network chexnet_frozen_graph_1541777429.pb --input_node input_tensor --output_nodes my_sigmoid_tensor --img_size 224 --img_file image.jpg --labellist labellist_chest_x_ray.json           