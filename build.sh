IMAGE_NAME=sanjayrancher/training-controller:v0.1-dev
docker build . -t $IMAGE_NAME -f ./Dockerfile

docker push $IMAGE_NAME
