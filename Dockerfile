# Dockerfile

FROM python:3.6.6

# install required debian packages
# add any package that is required after `python-dev`, end the line with \
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils\
    build-essential \
    python-dev \
    ibglib2.0-0 \
    cmake \
    telnet \
    vim \
&& rm -rf /var/lib/apt/lists/*

# install requirements
COPY src/requirements.txt /tmp/requirements.txt
COPY src/abc3.mp4 /tmp

RUN pip3 install -r /tmp/requirements.txt

# set /app as working directory
WORKDIR /app

# copy current directory to /app
COPY ./src/ /app
RUN rm -rf /app/.config

RUN rm -rf /app/known
RUN mkdir -p /app/known

# run python script
CMD ["python3", "camera.py"]

#docker run -d  -e HOST_PORT=5000 -p 5000:5000 camera-app
