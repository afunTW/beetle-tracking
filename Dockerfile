FROM afun/cuda:cuda9-cudnn7-py35
MAINTAINER afun
SHELL ["/bin/bash", "-c"]
WORKDIR /root/beetle-tracking

# Set locale
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install Python dependencies by pipenv
COPY . /root/beetle-tracking
RUN pipenv --python python3  && pipenv lock && pipenv sync
ENTRYPOINT pipenv shell
