# Training

NOTE: training is not needed if you only wish to evaluate PF-Track.

As described in the paper, training PF-Track involves two steps: (1) training a single-frame detector; (2) training a multi-frame tracker initialized from the detector. This procedure saves time and accelerates convergence.

**If you have downloaded our checkpoints, you can directly use our detector checkpoints and advance to step two.** However, please make sure the `load_from=xxx` fields in the configuration files point to the correct paths of checkpoints.

## 1. Training Procedure

### 1.1 Step One: Single-frame Training

You can skip this step if you have downloaded our pretrained single-frame model (see "[pretrained models](./pretrained.md)" for details).

* `CONFIG_PATH` will be your desired model configurations in [configs](./projects/configs/). In this repository, these are the files starting with `f1`.
* `GPU_NUM` is the number of GPUs. By default, I use 8.
* `PATH_TO_SAVE` is the directory to save the checkpoints.

The training command is as below:
```bash
bash tools/dist_train.sh CONFIG_PATH GPU_NUM --work-dir PATH_TO_SAVE
```

For example, if you want to train a small-resolution model, the command will be:
```bash
bash tools/dist_train.sh projects/configs/tracking/petr/f1_q5_800x320.py 8 --work-dir work_dirs/f1_pf_track/ 
```

The results should be similar to the single-frame detector checkpoints provided by us (see "[pretrained models](./pretrained.md)" for details).

### 1.2 Step Two: Multi-frame Training

* `CONFIG_PATH` will be your desired model configurations in [configs](./projects/configs/). In this repository, these are the files starting with `f3`.
* `GPU_NUM` is the number of GPUs. By default, I use 8.
* `PATH_TO_SAVE` is the directory to save the checkpoints.

The training command is as below:
```bash
bash tools/dist_train.sh CONFIG_PATH GPU_NUM --work-dir PATH_TO_SAVE
```

For example, if you want to train a small-resolution model, the command will be:
```bash
bash tools/dist_train.sh projects/configs/tracking/petr/f3_q5_800x320.py 8 --work-dir work_dirs/f3_pf_track/ 
```

The results should be similar to the `f3_all/final.pth` pretrained model provided by us.

## 2. Notes for Developers

* *Best practice for debugging?* We recommand setting `GPU_NUM=1` when running the above commands, and set `pdb.set_trace()` at the place you want to probe. Meanwhile, you can also switch the infos file to `tracking_forecasting-mini_infos_val.pkl` to save the time of data loading.
* *Core functions?* The core functions of tracking happens in [[code link](../projects/tracking_plugin/models/trackers/tracker.py)], the function `forward_tracking`. If you have any difficulty understanding my implementation, please read "[my designs](./designs.md)," from which you will learn about how to build an end-to-end tracking system.
* *Names for loss functions?* If you check the log, we use `loss_for` to represent motion forecasting losses in future reasoning, and `loss_mem` to represent past reasoning losses.