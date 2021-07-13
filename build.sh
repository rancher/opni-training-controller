IMAGE_NAME=tybalex/gpuservice-controller:dev
docker build . -t $IMAGE_NAME -f ./Dockerfile

docker push $IMAGE_NAME
