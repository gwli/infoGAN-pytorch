tag=nvdevtools/pytorch:0.2.0-cuda-8.0

nvidia-docker run --privileged=true  -e DISPLAY  --net=host --ipc=host -it --rm  -p 7022:22 -p 5022:5022 \
     -v $HOME/.Xauthority:/home/nvidia/.Xauthority \
     -v /raid/victor:/raid/victor \
     -v /raid/tools:/raid/tools \
     -w `pwd`  \
     $tag /bin/bash

