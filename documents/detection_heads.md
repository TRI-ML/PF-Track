# Integrating Various Detection Heads

PF-Track can cooperate with virtually any DETR-style 3D detection heads. The integration is simple and straightforward after my tutorials.

We believe this is meaningful because 3D detection is advancing fast recently, and detectors are actually the major influencer of MOTA-based metrics (which is kind of sad for tracking). But it is always good to incorporate state-of-the-art detectors and keep autonomous driving in its safest form.

## 1. Design

The key to working with any DETR-style heads is to provide the reference points, query features, and positional embeddings as the input to the decoder, instead of maintaining them as the internal states of detection heads, which is the common implementation of existing 3D detectors. Such a design is necessary because these information changes across frames and should be maintained by the tracker. 

Concretely, please look at our [code](../projects/tracking_plugin/models/trackers/tracker.py), where the detection head `pts_bbox_head` accepts the states of objects maintained in the `track_instances`:
```python
out = self.pts_bbox_head(img_feats[frame_idx], img_metas_single_frame, 
                         track_instances.query_feats, track_instan query_embeds, 
                         track_instances.reference_points)
```

Meanwhile, please compare the PETR head implemented in [our tracking](../projects/tracking_plugin/models/dense_heads/petr_tracking_head.py) and [original detector](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py) to have a more grounded sense.

## 2. Example with DETR3D Heads

In the paper, we have experiments with DETR3D heads in addition to the PETR heads. Please run our experiments for DETR3D by:
```bash
bash tools/dist_train.sh projects/configs/tracking/detr3d/f3_q500_1600x640.py 8 --work-dir word_dirs/debug/
``` 
NOTE: if you encounter `fp16` errors, un-annotate the `@auto_fp16(apply_to=('img'), out_fp32=True)` before `extract_feat` function in [tracker.py](../projects/tracking_plugin/models/trackers/tracker.py).

Comparing DETR3D with PETR, you will learn better about how to seamlessly integrate any DETR-style detector with PF-Track.