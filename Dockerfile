FROM python:3.10.9-slim

WORKDIR /usr/src/app

COPY . .

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

CMD [ "python", "main.py" ]