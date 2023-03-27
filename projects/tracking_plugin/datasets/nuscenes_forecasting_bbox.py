# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
from nuscenes.utils.data_classes import Box as NuScenesBox
from typing import Tuple, List, Dict
from pyquaternion import Quaternion
import numpy as np, copy


class NuScenesForecastingBox(NuScenesBox):
    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None,
                 forecasting: List[float] = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        :param: forecasting trajectories
        """
        super(NuScenesForecastingBox, self).__init__(center, size, orientation, label,
                                                     score, velocity, name, token)
        self.forecasting = forecasting
    
    def rotate(self, quaternion: Quaternion) -> None:
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)
        if self.forecasting is not None:
            self.forecasting = np.dot(quaternion.rotation_matrix[:2, :2], self.forecasting.T).T
    
    def copy(self) -> 'NuScenesForecastingBox':
        return copy.deepcopy(self)
