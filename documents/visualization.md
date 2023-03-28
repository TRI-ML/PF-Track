# Visualization Tools

Want to know how we made the [beautiful demos](https://www.youtube.com/watch?v=eJghONb2AGg)? Our programs are in `./tools/video_demo/`.

## 1. Multi-camera Tracking Video
* `INFO_FILE_PATH`: path to the infos file.
* `TRACKING_RESULT`: path to your tracking result file in `.json` format.
* `SHOW_DIR`: directory to save the images and videos.

The command is:
```bash
python tools/video_demo/cam_demo.py --data_infos_path INFO_FILE_PATH --result TRACKING_RESULT --show-dir SHOW_DIR
```

For example, if you want to visualize the results on the validation set of mini-split for `results.json`, please run the following commands:
```bash
python tools/video_demo/cam_demo.py --data_infos_path ./data/nuscenes/tracking_forecasting-mini_infos_val.pkl --result results.json --show-dir ./work_dirs/visualizations/
```

## 2. Bird's-eye-view (BEV) Video

By projecting the 3D bounding boxes to the BEV space, we can assess the quality of boxes more easily. Please remember to install `SimpleTrack` as "[environment set](./environment.md)" to use our BEV visualization program.

* `CONFIG_PATH`: configuration path.
* `TRACKING_RESULT`: path to your tracking result file in `.json` format.
* `SHOW_DIR`: directory to save the images and videos.

The command is:
```bash
python tools/video_demo/bev.py CONFIG_PATH --result TRACKING_RESULT --show-dir SHOW_DIR
```

For example, if you want to visualize the results on the validation set of mini-split for `results.json`, I will run the following commands:
```bash
python tools/video_demo/bev.py ./projects/configs/tracking/petr/f3_q500_800x320.py --result results.json --show-dir ./work_dirs/visualizations/
```

## 3. Notes for Developers

* *Point clouds.* In BEV visualization, we overlay the point clouds for visual clarity.
* *Color of boxes.* We assign a fixed color for each ID as `color = COLOR_MAP[COLOR_KEYS[track_id % len(COLOR_KEYS)]]` so that the consistency of colors reflect the quality of tracking.
* *Motion prediction.* In BEV visualization, we also visualize the predicted trajectories.