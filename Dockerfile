FROM python:3.5-slim

ENV DEBIAN_FRONTEND noninteractive

RUN apt update && \
    apt install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev && \
    mkdir opencv && \
    cd opencv && \
    git clone https://github.com/opencv/opencv.git && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ../opencv && \
    make && \
    make install

RUN cd ~ && \
    curl -L \
      http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2 \
      -o shape_predictor_68_face_landmarks.dat.bz2 && \
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2

RUN pip install --upgrade \
      opencv-python \
      numpy \
      dlib
