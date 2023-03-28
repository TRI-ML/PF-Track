# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from mmcv import Config
from mmdet3d.datasets import build_dataset
import matplotlib.pyplot as plt
from mot_3d.data_protos import BBox
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from mot_3d.visualization.visualizer2d import Visualizer2D


def parse_args():
    parser = argparse.ArgumentParser(description='3D Tracking Visualization')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--result', help='results file in pickle format')
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    dataset = build_dataset(cfg.data.visualization)
    results = json.load(open(args.result))['results']
    sample_tokens = results.keys()
    data_infos = dataset.data_infos
    data_info_sample_tokens = [info['token'] for info in data_infos]

    pbar = tqdm(total=len(results))
    for sample_idx, sample_token in enumerate(sample_tokens):
        # locate the information in data_infos
        data_info_idx = data_info_sample_tokens.index(sample_token)
        sample_info = data_infos[data_info_idx]
        raw_data = dataset[data_info_idx]
        
        # create location for visualization
        scene_token = sample_info['scene_token']
        seq_dir = os.path.join(args.show_dir, scene_token)
        os.makedirs(seq_dir, exist_ok=True)

        # get the point cloud information
        pc = raw_data['points'].data[0].numpy()[:, :3]
        mask = (np.max(pc, axis=-1) < 60)
        pc = pc[mask]

        l2e_r = sample_info['lidar2ego_rotation']
        l2e_t = sample_info['lidar2ego_translation']
        e2g_r = sample_info['ego2global_rotation']
        e2g_t = sample_info['ego2global_translation']
        l2e_r = Quaternion(l2e_r).rotation_matrix
        e2g_r = Quaternion(e2g_r).rotation_matrix
        l2e, e2g = np.eye(4), np.eye(4)
        l2e[:3, :3], l2e[:3, 3] = l2e_r, l2e_t
        e2g[:3, :3], e2g[:3, 3] = e2g_r, e2g_t
        l2g = e2g @ l2e
        new_pcs = np.concatenate((pc,
                                  np.ones(pc.shape[0])[:, np.newaxis]),
                                  axis=1)
        pc = ((new_pcs @ l2e.T) @ e2g.T)[:, :3]

        # gt_bboxes, instance_ids = sample_info['gt_boxes'], sample_info['instance_inds']
        visualizer = Visualizer2D(name=str(data_info_idx), figsize=(20, 20))
        COLOR_KEYS = list(visualizer.COLOR_MAP.keys())
        visualizer.handler_pc(pc)

        ego_xyz = l2g[:3, 3]
        plt.xlim((ego_xyz[0] - 60, ego_xyz[0] + 60))
        plt.ylim((ego_xyz[1] - 60, ego_xyz[1] + 60))
        # for i, (box, obj_id) in enumerate(zip(gt_bboxes, instance_ids)):
        #     bbox = BBox(x=box[0], y=box[1], z=box[2],
        #                 w=box[3], l=box[4], h=box[5],
        #                 o=-(box[6] + np.pi / 2))
        #     bbox = BBox.bbox2world(e2g @ l2e, bbox)
        #     visualizer.handler_box(bbox, linestyle='dashed', color='black')
        
        frame_results = results[sample_token]
        for i, box in enumerate(frame_results):
            if box['tracking_score'] < 0.4:
                continue
            nusc_box = Box(box['translation'], box['size'], Quaternion(box['rotation']))
            mot_bbox = BBox(
                x=nusc_box.center[0], y=nusc_box.center[1], z=nusc_box.center[2],
                w=nusc_box.wlh[0], l=nusc_box.wlh[1], h=nusc_box.wlh[2],
                o=nusc_box.orientation.yaw_pitch_roll[0]
            )
            track_id = int(box['tracking_id'].split('-')[-1])
            color = COLOR_KEYS[track_id % len(COLOR_KEYS)]
            visualizer.handler_box(mot_bbox, message='', color=color)
            # visualizer.handler_box(mot_bbox, message=box['tracking_id'].split('-')[-1], color=color)
        
        # forecasting prediction visualization
        all_trajs = list()
        color_list = list()
        for i, box in enumerate(frame_results):
            if box['tracking_score'] < 0.4:
                continue
            if 'forecasting' in box.keys() and box['forecasting'] is not None:
                # traj [T * 2]
                traj = np.asarray(box['forecasting'])
                traj = np.cumsum(traj, axis=0)
                # add a dummy zero beforehand
                traj = np.concatenate((np.zeros((1, 2)), traj), axis=0)
                traj += np.array(box['translation'])[:2]
                all_trajs.append(traj[np.newaxis, ...])

                track_id = int(box['tracking_id'].split('-')[-1])
                color = visualizer.COLOR_MAP[COLOR_KEYS[track_id % len(COLOR_KEYS)]]
                color_list.append(color)

        # if len(all_trajs) > 0:
        #     all_trajs = np.concatenate(all_trajs)
        #     traj_num, T, dim = all_trajs.shape
        #     new_trajs = all_trajs
        #     for i in range(traj_num):
        #         plt.plot(new_trajs[i, :, 0], new_trajs[i, :, 1], color=color_list[i])
        
        # forecasting gt visualization
        # if 'forecasting_locs' in sample_info.keys():
        #     trajs = sample_info['forecasting_locs'][:, :9, :]
        #     traj_num, ts, dim = trajs.shape
        #     new_trajs = trajs.reshape((traj_num * ts, dim))
        #     new_trajs = np.concatenate((new_trajs,
        #                                 np.ones(new_trajs.shape[0])[:, np.newaxis]),
        #                                 axis=1)
        #     new_trajs = ((new_trajs @ l2e.T) @ e2g.T)[:, :3].reshape((traj_num, ts, dim))
        #     for i in range(traj_num):
        #         plt.plot(new_trajs[i, :, 0], new_trajs[i, :, 1], color='green', linestyle='dashed')

        visualizer.save(os.path.join(seq_dir, f'{data_info_idx}.png'))
        visualizer.close()

        pbar.update(1)
    pbar.close()

    print('Making Videos')
    scene_tokens = os.listdir(args.show_dir)
    for video_index, scene_token in enumerate(scene_tokens):
        show_dir = os.path.join(args.show_dir, scene_token)
        fig_names = os.listdir(show_dir)
        indexes = sorted([int(fname.split('.')[0]) for fname in fig_names if fname.endswith('png')])
        fig_names = [f'{i}.png' for i in indexes]

        make_videos(show_dir, fig_names, 'video.mp4', show_dir)


def make_videos(fig_dir, fig_names, video_name, video_dir):
    import imageio
    import os
    import cv2

    fileList = list()
    for name in fig_names:
        fileList.append(os.path.join(fig_dir, name))

    writer = imageio.get_writer(os.path.join(video_dir, video_name), fps=2)
    for im in fileList:
        writer.append_data(cv2.resize(imageio.imread(im), (2000, 2000)))
    writer.close()
    return


if __name__ == '__main__':
    main()