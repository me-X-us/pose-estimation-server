FROM nvidia/cuda:11.0-base-ubuntu20.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
git \
python3-pip \
ffmpeg \
libsm6 \
libxext6 \
curl \
unzip \
wget \
&& rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/me-X-us/Pose-estimation.git
WORKDIR /Pose-estimation
RUN pip3 install -r requirements.txt && mkdir weights

WORKDIR /Pose-estimation/weights 
RUN curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38" > /dev/null && \
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38" -o pose_hrnet_w32_256x192.pth

WORKDIR /Pose-estimation/models/detectors/yolo
RUN pip3 install -r requirements.txt
WORKDIR /Pose-estimation/models/detectors/yolo/weights
RUN chmod 777 download_weights.sh && ./download_weights.sh

WORKDIR /Pose-estimation
ENTRYPOINT ["python3"]
CMD ["./scripts/server.py"]
