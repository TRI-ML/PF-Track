# An ABC Guide to End-to-end MOT

We almost doubted ourselves when developing PF-Track: how can it be **THAT** hard to change a detection head for end-to-end MOT? If you are or will have the same struggle, please put down your work and read this guide carefully. It could save your three months or white flags.

## 1. Principles of Design

### 1.1 Separate Training and Inference Settings. 

PF-Track first addresses the challenges in training and inference settings. Before delving into details, we would like to clarify the differences between **single-frame detection**, **multi-frame detection**, and **MOT**.

* Single-frame detection:
  * Description. Predicting bounding boxes from single-frame input.
  * Training. **Independent** samples comprising of single-frame input.
  * Inference. **Independent** samples comprising of single-frame input.
* Multi-frame detection:
  * Description. Predicting bounding boxes from $K$ frames of sensor inputs. Please note that $K$ is a fixed hyper-parameter.
  * Training. **Independent** samples comprising of video clips with $K$ frames.
  * Inference. **Independent** samples comprising of video clips with $K$ frames.
* MOT:
  * Description: Predicting tracks (bounding boxes and IDs) from an unbounded streaming sequence.
  * Training. **Independent** samples comprising of video clips with $K$ frames.
  * Inference. Iteratively predict boxes and IDs on continuous frames of a streaming sequence with unbounded length.

As clearly illustrated, the challenges of MOT lies in the gap between training/inference stages and corresponding differences in their designs. (MOT cannot train on streaming sequences because of limited GPU memoeires and indepdendent sampling of data loaders.)

In the PF-Track framework, we resolve such challenge from the aspects of data loading and model:
* Data loading: [nuscenes_tracking_dataset.py](../projects/tracking_plugin/datasets/nuscenes_tracking_dataset.py)
  * We shuffle the indexes of frames during training, but sample iteratively following the indexes during inference. Therefore, we keep the order of video sequences intact.
  * As in configuration files, we sample $K=3$ frames per training sample, but only sample $K=1$ frame per inference sample.
  * For temporal coherence, the data augmentation for MOT is also different. For example, we disable random-flip so that neighboring frames will not disagree with each other.
* Model: [tracker.py](../projects/tracking_plugin/models/trackers/tracker.py)
  * We design two interfaces for training and inference: `forward_train` and `forward_track`.
  * The tracking function maintains the internal states of tracks and examines if the video sequence has shifted.

### 1.2 Modular Detection, Past Reasoining, and Future Reasoning

MOT is a complex system and a challenge to your software enginnering skills. In PF-Trak, we emulate a modular design.

* [tracker.py](../projects/tracking_plugin/models/trackers/tracker.py) constructs a `Cam3DTracker` object to maintain the internal states of tracks, call the detection heads `self.pts_bbox_head(...)`, and conduct spatio-temporal reasoning.
* [spatial_temporal_reason.py](../projects/tracking_plugin/models/trackers/spatial_temporal_reason.py) is responsible for past and future reasoning. It receives the states of tracks (features, locations, etc.) as input and predicts refined bounding boxes, features, and future trajectories as output.
* We use `track_instances` to record per-track information in a unified way. It ws originally developed in `detectron2` and is a really handy tool when diverse information for an object is recorded. Please checkout `generate_empty_instance` in  [tracker.py](../projects/tracking_plugin/models/trackers/tracker.py) to learn about the states of tracks and `frame_shift` in [spatial_temporal_reason.py](../projects/tracking_plugin/models/trackers/spatial_temporal_reason.py) about how we manipulate `track_instances`.


## 2. Framework

### 2.1 Interfaces of Components and Functions

The best way to understand end-to-end 3D MOT is to read the function `forward_train` and `forward_track` in [tracker.py](../projects/tracking_plugin/models/trackers/tracker.py). We draw the sketch of modules and functions as below.

![code flow](../assets/code_flow.png)

* **Image features.** The function `extract_feat(...)` uses backbone CNNs to extract the feature maps of images.
* **Detection head.** Then we deode the bounding boxes, confidence scores from the image features and the states of tracks. Notably, we use the reference points, query features, and query positional embeddings.
```python
out = self.pts_bbox_head(img_feats[frame_idx], img_metas_single_frame, 
                         track_instances.query_feats, track_instan query_embeds, 
                         track_instances.reference_points)
```
* **Saving detection results.** We use `load_detection_output_into_cache` to record the detection results in the `track_instances`. As the detection results are not final, we save the detection results at a temporary buffer of `cache` fields, e.g., `cache_reference_points`.
* **Spatial-temporal Modeling.** Use the forward function as `track_instances = self.STReasoner(track_instances)` to conduct past and future reasoning. In `STReasoner`:
  * The function `frame_shift` moves the information in `hist` and `fut` fields to align with the current frame.
  * `forward_history_reasoning` enhances the query features and `forward_history_refine` generate refined bounding boxes and scores. The results are also saved in the `cache` fields.
  * `forward_future_reasoning` and `forward_future_prediction` forecasts the future trajectories.
* **Results on the current frame.** `frame_summarization` extracts the information from the `cache` fields and load them into the actual internal states of `track_instances`. For example, copying the `cache_reference_points` to `reference_points`. During the training stage, we update with all the `cache` information; while during the inference stage, only the objects with high scores are selected to update.
* **Propagate the queries.** We manipulate the reference points of objects by first compensating the motions with `update_reference_points` and then transform the cross-frame ego motions via `update_ego`.
* **Start a new frame.** We combine the tracks from the previous frame with the initialized detection queries via `next_frame_track_instances = Instances.cat([empty_track_instances, active_track_instances])`. Then the tracking process starts on the next frame.

The most notable design choice that simplifies my implementation is **introducing the cache fields**. It looks straighforward from a hindsight, but is actually not that easy to come up with.

### 2.2 Loss functions.

#### 2.2.1 Description

The loss functions play a critical role in tracking-by-attention because they supervise the queries to capture temporal coherent objects. The key idea is to assign the same object to the same ground truth across frames. Please look at the paper [MOTR](https://arxiv.org/abs/2105.03247) for an intuitive explanation. In our code, please check [tracking_loss.py](../projects/tracking_plugin/models/losses/tracking_loss.py), especially the parts of:
```python
for trk_idx in range(len(track_instances)):
    obj_id = track_instances.obj_idxes[trk_idx].item()
    if obj_id >= 0:
        if obj_id in obj_idx_to_gt_idx:
            track_instances.matched_gt_idxes[trk_idx] = obj_idx_to_gt_i[obj_id]
        else:
            num_disappear_track += 1
            track_instances.matched_gt_idxes[trk_idx] = -2
    else:
        track_instances.matched_gt_idxes[trk_idx] = -1
```

#### 2.2.2 Difference with MOTR

MOTR is a baseline tracking-by-attention project and we learn a lot from it. However, we would like to highlight an implementation difference compared to MOTR. Specifically, MOTR does not differentiate disappearing objects and emerging new objects, as in the code below. (Also at [here](https://github.com/megvii-research/MOTR/blob/8690da3392159635ca37c31975126acf40220724/models/motr.py#L201)).
```python
for trk_idx in range(len(track_instances)):
    obj_id = track_instances.obj_idxes[trk_idx].item()
    if obj_id >= 0:
        if obj_id in obj_idx_to_gt_idx:
            track_instances.matched_gt_idxes[trk_idx] = obj_idx_to_gt_i[obj_id]
        else:
            num_disappear_track += 1
            # Pay attention to the line below.
            track_instances.matched_gt_idxes[trk_idx] = -1
    else:
        track_instances.matched_gt_idxes[trk_idx] = -1
```
This implies that we will not match disappearing objects with other GTs, unlike MOTR. we think it makes more sense.

### 2.3 Special Operations for Inference

While reading our code, please pay attention to how `forward_track` is different from `forward_train`.

1. *Deal with changing sequences.* During the inference stage, the program tells if the incoming frame comes from a new video sequence, which implies the need for refreshing the `RunTimeTracker` and `track_instances`.

```python
if self.runtime_tracker.timestamp is None or abs(timestamp[0] - self.runtime_tracker.timestamp) > 10:
    self.runtime_tracker.timestamp = timestamp[0]
    self.runtime_tracker.current_seq += 1
    self.runtime_tracker.track_instances = None
    self.runtime_tracker.current_id = 0
    self.runtime_tracker.l2g = None
    self.runtime_tracker.time_delta = 0
    self.runtime_tracker.frame_index = 0
self.runtime_tracker.time_delta = timestamp[0] - self.runtime_tracker.timestamp
self.runtime_tracker.timestamp = timestamp[0]
```

2. *Record tracking instances inside runtime tracker*.

3. *Select tracking categories.* In nuScenes, the 3D MOT challenge only concerns a subset of 7 categories in detection, leaving out the static objects like traffic cones. Therefore, we have the operations below to select the related objects for output.
```python
max_cat = torch.argmax(out['all_cls_scores'][-1, 0, :].sigmoid(), dim=-1)
related_cat_mask = (max_cat < 7) 
track_instances = track_instances[related_cat_mask]
```

4. *Assign IDs to tracks.* 
```python
active_mask = (track_instances.scores > self.runtime_tracker.threshold)
for i in range(len(track_instances)):
    if track_instances.obj_idxes[i] < 0:
        track_instances.obj_idxes[i] = self.runtime_tracker.current_id 
        self.runtime_tracker.current_id += 1
        if active_mask[i]:
            track_instances.track_query_mask[i] = True
out['track_instances'] = track_instances
```

5. *Filter by scores before output*. We only output the bounding boxes with scores larger than a certain threshold `output_threshold=0.2`:
```python
score_mask = (track_instances.scores > self.runtime_tracker.output_threshold)
out['all_masks'] = score_mask.clone()
```

6. *Training stage does not concern life cycles.* Please check our `frame_summarization`:
```python
# inference mode
if tracking:
    active_mask = (track_instances.cache_scores >= self.runtime_track.record_threshold)
# training mode
else:
    track_instances.bboxes = track_instances.cache_bboxes.clone()
    track_instances.logits = track_instances.cache_logits.clone()
    track_instances.scores = track_instances.cache_scores.clone()
    active_mask = (track_instances.cache_scores >= 0.0)
```