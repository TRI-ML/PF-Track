# Environment Setup

Our implementation mainly relies on `mmdetection3d==v0.17.1`, the same as previous vision-based 3D detection methods. For detailed instructions, please refer to the official documentations of `mmdetection3d`.

However, we understand the pain for installation, because many `mmlab` modules need to be built from source and the version mismatches can cause confusing error messages. 
1. We recommend that you directly use our [docker file](./docker/Dockerfile-pftrack).

If you are new to docker, I recommand the following bash scipts.
```bash
# Remember to update the DATA_ROOT and CKPTS_ROOT in Makefile to your own paths.
# Delete the AWS and WANDB options in Makefile if you don't use them.
make docker-build
make docker-dev
cd ../PF_Track
```

2. If docker does not work for you, you may manually install the required packages referring to our [docker file](./docker/Dockerfile-pftrack), then go to step 3. For your convenience, some important versions of our packages are: 

* Pytorch-related. `pytorch==1.9.0`, `cuda==11.1`, `cudnn==8`.
* MMLab-related. (build from source recommended). `MMCV==1.4.0`, `MMDetection==v2.24.1`, `MMSegmentation==v0.20.2`, `MMDetection3d==v0.17.1`.

3. Please also install these dependencies manally with `pip`.
* nuScenes-related. `nuscenes-devkit==1.1.7`, `motmetrics==1.1.3` (don't use higher versions of `motmetrics`, or it will cause `nuscenes-devkit` into bugs, which is a known nuScenes dependency issue.)
* Visualization-tools. (optional). Install [SimpleTrack](https://github.com/tusen-ai/SimpleTrack) locally via `pip install -e ROOT_OF_SIMPLETRACK` for BEV visualization.

4. We thank [DETR3D](https://github.com/WangYueFt/detr3d) and [PETR](https://github.com/megvii-research/PETR) for open-sourcing. The `projects/mmdet3d_plugin/` directory are the basic modules for 3D detection insipired by their implementations. We have already contained them for your convenience.