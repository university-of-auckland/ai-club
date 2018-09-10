FROM tensorflow/tensorflow:latest

WORKDIR /aiuoa
RUN pip install tflearn pathlib

CMD /bin/bash
