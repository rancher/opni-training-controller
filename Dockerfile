FROM nikolaik/python-nodejs:python3.8-nodejs14-slim
WORKDIR /app
COPY ./training_controller/ /app/
RUN chmod a+rwx -R /app
RUN pip install --no-cache-dir -r requirements.txt
RUN npm install elasticdump -g
CMD ["python", "main.py"]
