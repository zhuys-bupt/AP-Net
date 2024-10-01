from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .eht3d_dataset import ETH3D

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "eth3d": ETH3D
}
