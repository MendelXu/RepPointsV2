import numpy as np

import torch

from mmdet.core import bbox2result, bbox_mapping_back, multiclass_nms
from ..builder import DETECTORS
from .reppoints_v2_detector import RepPointsV2Detector


@DETECTORS.register_module()
class RepPointsV2LightDetector(RepPointsV2Detector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RepPointsV2LightDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                       test_cfg, pretrained)
        print(self)

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
