# Modified from SimpleTrack (https://github.com/tusen-ai/SimpleTrack)
from threading import local
import matplotlib.pyplot as plt, numpy as np
from . import functions
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
import copy


class Visualizer2D:
    def __init__(self, name='', figsize=(8, 8)):
        self.figure = plt.figure(name, figsize=figsize)
        plt.axis('equal')
        self.COLOR_MAP = {
            'gray': np.array([140, 140, 136]) / 256,
            'light_blue': np.array([4, 157, 217]) / 256,
            'red': np.array([191, 4, 54]) / 256,
            'black': np.array([0, 0, 0]) / 256,
            'purple': np.array([224, 133, 250]) / 256, 
            'dark_green': np.array([32, 64, 40]) / 256,
            'green': np.array([77, 115, 67]) / 256
        }
    
    def show(self):
        plt.show()
    
    def close(self):
        plt.close()
    
    def save(self, path):
        plt.savefig(path)
    
    def handler_pc(self, pc, color='gray'):
        vis_pc = np.asarray(pc)
        plt.scatter(vis_pc[:, 0], vis_pc[:, 1], marker='o', color=self.COLOR_MAP[color], s=0.01)
    
    def handle_project_pc(self, pc, image, ego_pose, cam_pose, cam_intrinsics):
        points, coloring = functions.map_point_cloud_to_image(
            pc, image, ego_pose, cam_pose, cam_intrinsics)
        plt.scatter(points[:, 0], points[:, 1], c=coloring, s=0.1)
    
    def handle_image(self, image):
        plt.imshow(image)
    
    def handle_bbox(self, bbox: Box, message: str='', color='red', linestyle='solid'):
        """bbox bev visualization
        """
        corners = bbox.bottom_corners().T
        corners = np.concatenate([corners, corners[0:1, :2]])
        plt.plot(corners[:, 0], corners[:, 1], color=self.COLOR_MAP[color], linestyle=linestyle)
        corner_index = np.random.randint(0, 4, 1)
        plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=self.COLOR_MAP[color])

    def handle_project_bbox(self, bbox: Box, image, ego_pose, cam_pose, cam_intrinsics,
                            message: str='', color='red', linestyle='solid'):
        """bbox project to image visualization
        """
        # transform global bbox to ego --> camera
        box = copy.deepcopy(bbox)
        box.translate(-ego_pose[:3, 3])
        box.rotate(Quaternion(matrix=ego_pose[:3, :3].T))
        box.translate(-cam_pose[:3, 3])
        box.rotate(Quaternion(matrix=cam_pose[:3, :3].T))

        if box.center[2] < 0:
            return

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                plt.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linestyle=linestyle)
                prev = corner

        corners = view_points(box.corners(), cam_intrinsics, normalize=True).T[:, :2]
        
        # if np.max(corners[:, 0]) <= 1 or np.min(corners[:, 1]) >= image.size[0] or \
        #     np.max(corners[:, 1]) <= 1 or np.min(corners[:, 1]) >= image.size[1]:
        #     return
        if not (np.min(corners[:, 0]) >= 1 and np.max(corners[:, 0]) <= image.size[0] and \
            np.min(corners[:, 1]) >= 1 and np.max(corners[:, 1]) <= image.size[1]):
            return

        for i in range(4):
            plt.plot([corners[i][0], corners[i + 4][0]],
                     [corners[i][1], corners[i + 4][1]],
                      color=self.COLOR_MAP[color], linestyle=linestyle)
        
        draw_rect(corners[:4], self.COLOR_MAP[color])
        draw_rect(corners[4:], self.COLOR_MAP[color])
        
        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners[[2, 3, 7, 6]], axis=0)
        plt.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=self.COLOR_MAP[color], linestyle=linestyle)
        
        # select a corner and plot messages
        corner_index = np.random.randint(0, 8, 1)
        plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=self.COLOR_MAP[color])
