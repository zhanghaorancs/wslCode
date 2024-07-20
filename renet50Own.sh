#!/bin/bash

# Run server.py in a new terminal
/home/zhang/anaconda3/envs/paperCode/bin/python /home/zhang/code/resnet50Test/resnet50OwnServer.py &
# /home/zhang/anaconda3/envs/paperCode/bin/python /home/zhang/code/resnet50Test/resnet50OwnServer2.py &
# Run worker1.py in a new terminal
/home/zhang/anaconda3/envs/paperCode/bin/python /home/zhang/code/resnet50Test/worker1.py &

/home/zhang/anaconda3/envs/paperCode/bin/python /home/zhang/code/resnet50Test/worker2.py &
/home/zhang/anaconda3/envs/paperCode/bin/python /home/zhang/code/resnet50Test/worker3.py &
/home/zhang/anaconda3/envs/paperCode/bin/python /home/zhang/code/resnet50Test/worker4.py &
# Run worker2.py in a new terminal
# gnome-terminal -- bash -c "python3 /path/to/worker2.py; exec bash"

wait