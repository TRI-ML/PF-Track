# Preprocessing nuScenes

## 1. Download nuScenes

Download the nuScenes dataset, soft-link it to `./data/nuscenes`. This step is compulsory.

## 2. Creating infos file

We follow the practice in `MMDetection3D` and create information files for training/evaluation the tracking and motion prediction tasks for PF-Track. You can either use our provided files or optionally create the infos file on your own.

### 2.1 Using Our Provided Infos File.

If you haven't downloaded the provided files, checkout "[pretrained models and data files](./pretrained.md)." If you have finished downloading, copy the infos file into `./data/nuscenes/`.

### 2.2 Reproducing the infos files. (Optional)

For training and validation splits, commands below generate `tracking_forecasting_infos_train.pkl` and `tracking_forecasting_infos_val.pkl`. (Around 3 hours)

```bash
python tools/create_data.py nuscenes-tracking --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag tracking_forecasting --version v1.0 --forecasting
```

For mini splits, commands below generate `tracking_forecasting-mini_infos_train.pkl` and `tracking_forecasting-mini_infos_val.pkl`. (Around 3 minutes)

```bash
python tools/create_data.py nuscenes-tracking --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag tracking_forecasting --version v1.0-mini --forecasting
```

For the test split, commands below generate `tracking_forecasting_infos_test.pkl`.

```bash
python tools/create_data.py nuscenes-tracking-test --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag tracking_forecasting --version v1.0 --forecasting
```

## 3. Notes for Developers

* *Core code?* We change detection-style information file (mmdetection3d) into a tracking-style information file (PF-track) mainly from the following aspects. [[Related Code 1](./tools/create_data.py)][[Related Code 2](./tools/data_converter/nuscenes_tracking_converter.py)][[Related Code 3](./tools/data_converter/nuscenes_prediction_tools.py)].
* *How to support tracking?* We load the ids of objects into the information files to form tracks across frames.
* *How to support prediction?* We load the future trajectories (13 frames, or ~6.0 seconds) to support the training of motion prediction.