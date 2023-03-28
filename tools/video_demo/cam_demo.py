# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
import mmcv
import os
import argparse
from nuscenes.nuscenes import NuScenes
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from typing import Tuple, List, Iterable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.data_classes import DetectionBox
from projects.tracking_plugin.visualization import NuscenesTrackingBox as TrackingBox
# from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.render import visualize_sample


COLOR_MAP = {
    'red': np.array([191, 4, 54]) / 256,
    'light_blue': np.array([4, 157, 217]) / 256,
    'black': np.array([0, 0, 0]) / 256,
    'gray': np.array([140, 140, 136]) / 256,
    'purple': np.array([224, 133, 250]) / 256, 
    'dark_green': np.array([32, 64, 40]) / 256,
    'green': np.array([77, 115, 67]) / 256,
    'brown': np.array([164, 103, 80]) / 256,
    'light_green': np.array([135, 206, 191]) / 256,
    'orange': np.array([229, 116, 57]) / 256,
}
COLOR_KEYS = list(COLOR_MAP.keys())


cams = ['CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT']

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from matplotlib import rcParams


def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None
                       ):
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #    boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #    boxes = nusc.get_boxes(sample_data_token)
    boxes = pred_anns
    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def render_sample_data(
        sample_toekn: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax=None,
        nsweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
        pred_data=None,
      ) -> None:
    # lidiar_render(sample_toekn, pred_data, out_path=out_path)
    sample = nusc.get('sample', sample_toekn)
    # sample = data['results'][sample_token_list[0]][0]
    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]
    if ax is None:
        _, ax = plt.subplots(2, 3, figsize=(24, 18))
    j = 0
    for ind, cam in enumerate(cams):
        sample_data_token = sample['data'][cam]

        sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            assert False
        elif sensor_modality == 'camera':
            # Load boxes and image.
            boxes = [TrackingBox(
                         'predicted',
                         record['translation'], record['size'], Quaternion(record['rotation']),
                         tracking_id=record['tracking_id'].split('-')[-1]) for record in
                     pred_data['results'][sample_toekn] if record['tracking_score'] > 0.4]

            data_path, boxes_pred, camera_intrinsic = get_predicted_data(sample_data_token,
                                                                         box_vis_level=box_vis_level, pred_anns=boxes)
            # _, boxes_gt, _ = nusc.get_sample_data(sample_data_token, box_vis_level=box_vis_level)
            if ind == 3:
                j += 1
            ind = ind % 3
            data = Image.open(data_path)

            # Show image.
            ax[j, ind].imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes_pred:
                    # c = np.array(get_color(box.name)) / 255.0
                    track_id = int(box.tracking_id)
                    color = COLOR_MAP[COLOR_KEYS[track_id % len(COLOR_KEYS)]]
                    box.render(ax[j, ind], view=camera_intrinsic, normalize=True, \
                        colors=color, linestyle='dashed', linewidth=1.5, text=False)

            # Limit visible range.
            ax[j, ind].set_xlim(0, data.size[0])
            ax[j, ind].set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax[j, ind].axis('off')
        ax[j, ind].set_aspect('equal')

    if out_path is not None:
        plt.savefig(out_path+'_camera', bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()
    plt.close()


def make_videos(fig_dir, fig_names, video_name, video_dir):
    import imageio
    import os
    import cv2

    fileList = list()
    for name in fig_names:
        fileList.append(os.path.join(fig_dir, name))

    writer = imageio.get_writer(os.path.join(video_dir, video_name), fps=2)
    for im in fileList:
        writer.append_data(cv2.resize(imageio.imread(im), (4000, 2800)))
    writer.close()
    return


def parse_args():
    parser = argparse.ArgumentParser(description='3D Tracking Visualization')
    parser.add_argument('--data_infos_path', type=str, default='./data/nuscenes/tracking_forecasting-mini_infos_val.pkl')
    parser.add_argument('--result', help='results file')
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data_infos = mmcv.load(args.data_infos_path)['infos']
    data_info_sample_tokens = [info['token'] for info in data_infos]

    nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes/', verbose=True)
    results = mmcv.load(args.result)
    sample_token_list = list(results['results'].keys())

    pbar = tqdm(total=len(sample_token_list))
    for i, sample_token in enumerate(sample_token_list):
        # prepare the directory for visualization
        data_info_idx = data_info_sample_tokens.index(sample_token)
        sample_info = data_infos[data_info_idx]
        scene_token = sample_info['scene_token']
        seq_dir = os.path.join(args.show_dir, scene_token)
        os.makedirs(seq_dir, exist_ok=True)
        out_path = os.path.join(seq_dir, f'{i}')

        # render
        render_sample_data(sample_token, pred_data=results, out_path=out_path)
        pbar.update(1)
    pbar.close()

    print('Making Videos')
    scene_tokens = os.listdir(args.show_dir)
    for video_index, scene_token in enumerate(scene_tokens):
        show_dir = os.path.join(args.show_dir, scene_token)
        fig_names = os.listdir(show_dir)
        indexes = sorted([int(fname.split('_')[0]) for fname in fig_names if fname.endswith('png')])
        fig_names = [f'{i}_camera.png' for i in indexes]

        make_videos(show_dir, fig_names, 'video.mp4', show_dir)