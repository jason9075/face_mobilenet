FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN pip install opencv-python xmltodict efficientnet tensorflow-addons==0.8.1 sklearn
RUN apt-get -y install libsm6 libxrender1 libxext-dev