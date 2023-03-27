# Inference

We explain the inference procedure on the validation set as examples.

## 1. Inference Commands.
* `CONFIG_PATH` will be your desired model configurations in [configs](./projects/configs/). It specifies the model architecture.
* `CHECK_POINT` will be the checkpoint of a trained model.
* `PATH_TO_SAVE` is the directory to save the 3D MOT inference results and evaluation metrics.

```bash
python tools/test_tracking.py CONFIG_PATH CHECK_POINT --jsonfile_prefix PATH_TO_SAVE  --eval bbox 
```

For example, we can evaluate a PF-Track model designed for $800\times 320$ resolution (small-resolution setting), trained at location `./work_dir/f3_petr_800x320/final.pth`, intended to save in `./work_dir/f3_petr_800x320/results/` via the following commands.

```bash
python tools/test_tracking.py projects/conf
igs/tracking/petr/f3_q500_800x320.py ./work_dir/f3_petr_800x320/final.pth --jsonfile_prefix ./work_dir/f3_petr_800x320/results --eval bbox
```
You can use the checkpoint provided by us for a quick try.

## 2. Notes for Developers.

* *Multi-GPU inference?* 3D MOT requires running sequentially on all the frames of nuScenes. Therefore, supporting distributed inference is not straightforward and we does not concern it currently.
* *Configuration files?* 
  * Pay attention to the fields of `test_tracking` fields in configuration files.
  * During the inference time, pay attention to `runtime_tracker` in the configurations. `score_threshold` controls the minimum detection score for output, `record_threshold` is the score threshold for using track extension, and `max_age_since_update` is both the length for track extension  and maximum age for a track before termination.
* *Core code?* The core functions of tracking happens in [[code link](../projects/tracking_plugin/models/trackers/tracker.py)], the function `forward_tracking`. If you have any difficulty understanding my implementation, please read "[my designs](./designs.md)," from which you will learn about how to build an end-to-end tracking system.