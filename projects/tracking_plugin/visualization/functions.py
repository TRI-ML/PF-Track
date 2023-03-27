
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
import copy
from matplotlib.axes import Axes
from typing import Tuple, List, Iterable


def map_point_cloud_to_image(pc, image, ego_pose, cam_pose, cam_intrinsics, min_dist=1.0):
    """ map a global coordinate point cloud to image
    Args:
        pc (numpy.ndarray [N * 3])
    """
    point_cloud = copy.deepcopy(pc)
    
    # transform point cloud to the ego
    point_cloud -= ego_pose[:3, 3]
    point_cloud = point_cloud @ ego_pose[:3, :3]

    # transform from ego to camera
    point_cloud -= cam_pose[:3, 3]
    point_cloud = point_cloud @ cam_pose[:3, :3]

    # project points to images
    # step 1. Depth and colors
    depths = point_cloud[:, 2]
    intensities = point_cloud[:, 2]
    intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
    intensities = intensities ** 0.1
    intensities = np.maximum(0, intensities - 0.5)
    coloring = intensities

    # step 2. Project onto images with intrinsics
    points = point_cloud.T
    points = view_points(points[:3, :], cam_intrinsics, normalize=True).T

    # step 3. Remove the points that are outside/behind the camera
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[:, 0] > 1)
    mask = np.logical_and(mask, points[:, 0] < image.size[0] - 1)
    mask = np.logical_and(mask, points[:, 1] > 1)
    mask = np.logical_and(mask, points[:, 1] < image.size[1] - 1)
    points = points[mask, :]
    coloring = coloring[mask]

    return points, coloring


class NuscenesTrackingBox(Box):
    """ Data class used during tracking evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 tracking_id: str = '',  # Instance id of this object.
                 tracking_name: str = '',  # The class name used in the tracking challenge.
                 tracking_score: float = -1.0):  # Does not apply to GT.

        super().__init__(translation, size, rotation, np.nan, tracking_score, name=tracking_id)

        assert tracking_name is not None, 'Error: tracking_name cannot be empty!'

        assert type(tracking_score) == float, 'Error: tracking_score must be a float!'
        assert not np.any(np.isnan(tracking_score)), 'Error: tracking_score may not be NaN!'

        # Assign.
        self.tracking_id = tracking_id
        self.tracking_name = tracking_name
        self.tracking_score = tracking_score

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.tracking_id == other.tracking_id and
                self.tracking_name == other.tracking_name and
                self.tracking_score == other.tracking_score)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'tracking_id': self.tracking_id,
            'tracking_name': self.tracking_name,
            'tracking_score': self.tracking_score
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   tracking_id=content['tracking_id'],
                   tracking_name=content['tracking_name'],
                   tracking_score=-1.0 if 'tracking_score' not in content else float(content['tracking_score']))
    
    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linestyle: str = 'solid',
               linewidth: float = 2,
               text=True) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth, linestyle=linestyle)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors, linewidth=linewidth, linestyle=linestyle)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors)
        draw_rect(corners.T[4:], colors)

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors, linewidth=linewidth, linestyle=linestyle)
        corner_index = np.random.randint(0, 8, 1)
        if text:
            axis.text(corners[0, corner_index] - 1, corners[1, corner_index] - 1, self.tracking_id, color=colors, fontsize=8)
