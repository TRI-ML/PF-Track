WORK_DIR=${PWD}
PROJECT=PF_Track
DOCKER_IMAGE=${PROJECT}:latest
DOCKER_FILE=docker/Dockerfile-pftrack
DATA_ROOT=/mnt/fsx/datasets
CKPTS_ROOT=/mnt/fsx/ckpts

DOCKER_OPTS = \
	-it \
	--rm \
	-e DISPLAY=${DISPLAY} \
	-v /tmp:/tmp \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /mnt/fsx:/mnt/fsx \
	-v ~/.ssh:/root/.ssh \
	-v ~/.aws:/root/.aws \
	-v ${WORK_DIR}:/workspace/${PROJECT} \
	-v ${DATA_ROOT}:/workspace/${PROJECT}/data \
	-v ${CKPTS_ROOT}:/workspace/${PROJECT}/ckpts \
	--shm-size=1G \
	--ipc=host \
	--network=host \
	--pid=host \
	--privileged

DOCKER_BUILD_ARGS = \
	--build-arg AWS_ACCESS_KEY_ID \
	--build-arg AWS_SECRET_ACCESS_KEY \
	--build-arg AWS_DEFAULT_REGION \
	--build-arg WANDB_ENTITY \
	--build-arg WANDB_API_KEY \

docker-build:
	nvidia-docker image build -f $(DOCKER_FILE) -t $(DOCKER_IMAGE) \
	$(DOCKER_BUILD_ARGS) .

docker-dev:
	nvidia-docker run --name $(PROJECT) \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE) bash

clean:
	find . -name '"*.pyc' | xargs sudo rm -f && \
	find . -name '__pycache__' | xargs sudo rm -rf