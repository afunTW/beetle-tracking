FROM afun/cuda:cuda9-cudnn7-py35-py27
LABEL maintainer=afun@afun.tw
SHELL ["/bin/bash", "-c"]
WORKDIR /root/project

# Set locale
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . /root/project
RUN /bin/bash download_data.bash
RUN apt-get install -y libglib2.0-0 libsm6 libxrender-dev
RUN /bin/bash download_data.bash
RUN /bin/bash /root/project/setup.bash