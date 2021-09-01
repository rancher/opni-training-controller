FROM rancher/opni-python-base:3.8-node14-elasticdump
ENV NODE_TLS_REJECT_UNAUTHORIZED 0

WORKDIR /app
COPY ./training_controller/ /app/
RUN chmod a+rwx -R /app

CMD ["python", "main.py"]