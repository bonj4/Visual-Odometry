FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    python3.7.3 \
    python3-pip

WORKDIR /code

# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg\
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
    

# RUN /usr/local/bin/python -m pip install --upgrade pip

COPY ./requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

