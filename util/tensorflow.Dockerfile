FROM bamos/openface:latest

RUN pip install --upgrade pip setuptools

WORKDIR /aiuoa
RUN chmod -R a+rwx /aiuoa
RUN pip install tflearn pathlib tensorflow jyupter scikit-learn face_recognition
ENV PYTHONPATH "${PYTHONPATH}:/aiuoa"
EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/aiuoa/notebooks --ip 0.0.0.0 --no-browser --allow-root"]