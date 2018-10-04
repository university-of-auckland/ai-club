FROM python:3

RUN apt-get update
RUN apt-get install graphviz
RUN pip install --upgrade pip setuptools
RUN python --version

WORKDIR /aiuoa
RUN chmod -R a+rwx /aiuoa
RUN pip install -U tflearn
RUN pip install -U pathlib
RUN pip install -U tensorflow
RUN pip install -U jupyter
RUN pip install -U scikit-learn
RUN pip install -U face_recognition
RUN pip install -U pandas
RUN pip install -U seaborn
RUN pip install -U matplotlib
RUN pip install -U h5py
RUN pip install -U keras
RUN pip install -U graphviz
RUN pip install -U ann_visualizer

ENV PYTHONPATH "${PYTHONPATH}:/aiuoa"
RUN mkdir /aiuoa/.jupyter
ENV JUPYTER_CONFIG_DIR /aiuoa/.jupyter
EXPOSE 8888
EXPOSE 6006

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/aiuoa/notebooks --ip 0.0.0.0 --no-browser --allow-root"]
