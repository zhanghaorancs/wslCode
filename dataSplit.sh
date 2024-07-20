#!/bin/bash

/home/zhang/anaconda3/envs/paperCode/bin/python /home/zhang/code/dataSplit/dataSplitServer.py &
/home/zhang/anaconda3/envs/paperCode/bin/python /home/zhang/code/dataSplit/worker01.py &
/home/zhang/anaconda3/envs/paperCode/bin/python /home/zhang/code/dataSplit/worker02.py &
/home/zhang/anaconda3/envs/paperCode/bin/python /home/zhang/code/dataSplit/worker03.py &
/home/zhang/anaconda3/envs/paperCode/bin/python /home/zhang/code/dataSplit/worker03.py &

wait
