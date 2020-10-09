import numpy as np

import torch

from mmdet.core import bbox2result, bbox_mapping_back, multiclass_nms
from ..builder import DETECTORS
from .reppoints_detector import RepPointsDetector

@DETECTORS.register_module()
class RepPointsLightDetector(RepPointsDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RepPointsLightDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                       test_cfg, pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            feat = self.neck(x)
            feat = (x[0],)+feat
        else:
            feat = x
        return feat
