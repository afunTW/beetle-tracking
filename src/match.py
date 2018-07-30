import logging

import numpy as np

from munkres import Munkres
from src.utils import func_profile

LOGGER = logging.getLogger(__name__)


def _calc_cost_matrix_grade(l, w):
    return np.sum(np.array(l) * np.array(w))

def hungarian_matching(frame1, frame2):
    cost_matrix = np.zeros((frame1.bbox_count, frame2.bbox_count))

    for bid1, bbox1 in enumerate(frame1.bboxes):
        for bid2, bbox2 in enumerate(frame2.bboxes):
            iou = bbox1.calc_iou(bbox2)
            grade = np.average(np.array((bbox1.confidence, bbox2.confidence))) * iou
            cost_matrix[bid1, bid2] = grade
    
    m = Munkres()
    indexes = m.compute((-np.array(cost_matrix)).tolist())
    
    return indexes, cost_matrix