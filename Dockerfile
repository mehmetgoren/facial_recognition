FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update && apt-get upgrade -y
RUN apt-get install -y apt-utils
RUN apt-get install -y tzdata
RUN apt-get install -y curl
RUN apt-get -qq install --no-install-recommends -y python3-pip

RUN pip3 install addict
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install Pillow
RUN pip3 install psutil
RUN pip3 install redis
RUN pip3 install requests
RUN pip3 install scikit-learn
RUN pip3 install setuptools
RUN pip3 install tqdm==4.64.0

COPY . .

CMD ["python3", "main_pytorch.py"]