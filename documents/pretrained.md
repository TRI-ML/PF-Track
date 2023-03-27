# Pretrained Models and Data Files

We provide all the information needed for PF-Track experiments in [link](). 

## 1. Preprocessed nuScenes Files [[Download](https://tri-ml-public.s3.amazonaws.com/github/pftrack/data.zip)]

These files are the information files for data loading following the styles of `mmdetection3d`. Please download and extract the zip to your directory of nuScenes dataset. After doing so, the structure of that directory will be:
```txt
- nuscenes
    - v1.0-mini
    - v1.0-test
    - v1.0-trainval
    ...
    - tracking_forecasting_infos_test.pkl
    - tracking_forecasting_infos_train.pkl
    - tracking_forecasting_infos_val.pkl
    - tracking_forecasting-mini_infos_train.pkl
    - tracking_forecasting-mini_infos_val.pkl
```

FYI, you can also skip this step and reproduce our information files as "[preprocessing nuscenes](./preprocessing.md)," though it could take longer time.

## 2. Pretrained PF-Track Models [[Download](https://tri-ml-public.s3.amazonaws.com/github/pftrack/PF-Track-Models.zip)]

We provide the pretrained PF-Track models. After extracting the zip files, you will find `f3_all` as the small-resolution ($800\times 320$) PF-Track model and `f3_fullres_all` as the full-resolution model ($1600\times 640$). Please note that the inferenced tracking results in `.json` format, metric values on nuScenes, and training logs are also here.

## 3. Single-frame Detection Model [[Download](https://tri-ml-public.s3.amazonaws.com/github/pftrack/f1.zip)] (Optional) 

After downloading, please extract them to `./ckpts/` of the root directory of this repository. We provide the single-frame detectors from the first stage of training PF-Track. Downloading them will save the effort in training single-frame detectors for reproducing our results.

## 4. Pretrained Models for Ablation Studies [[Download](https://tri-ml-public.s3.amazonaws.com/github/pftrack/ablation.zip)] (Optional)

If you are interested in our implementation details, please use these checkpoints or tracking results to inspect our ablation studies. We mark the files by their index in our Table 2.