FROM rancher/opni-python-base:3.8-node14-elasticdump
ENV NODE_TLS_REJECT_UNAUTHORIZED 0

COPY ./training_controller/ /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod a+rwx -R /app

CMD ["python3", "./main.py"]
