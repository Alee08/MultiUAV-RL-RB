FROM ubuntu:18.04

MAINTAINER Alessandro Trapasso "ale.trapasso8@gmail.com"

ENV DEBIAN_FRONTEND="noninteractive"

# the first thing we specify in a Dockerfile is the base image
RUN apt-get update && apt-get -y install graphviz 
RUN apt-get install -y python3.6 python3-pip python3-dev \
    bzip2 wget jupyter 
RUN apt-get -y install python3-tk #Per il plot, non usato

# Install required packages
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt

# Install required python packages
RUN pip3 install -Ir  requirements.txt
#RUN apt-get install graphviz

COPY . /app


 # Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Command that starts up the notebook 
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

