import logging
from collections import Counter

import numpy as np

import cv2


class BBox(object):
    def __init__(self, frame_idx, pt1, pt2, confidence, multiclass_result):
        # attribute
        self._multiclass_result = multiclass_result
        self.frame_idx = frame_idx
        self.pt1, self.pt2 = pt1, pt2
        self.w, self.h = int(pt2[0] - pt1[0]), int(pt2[1] - pt1[1])
        self.center = tuple((int(pt2[0] + pt1[0]) // 2, int(pt2[1] + pt1[1]) // 2))
        self.confidence = confidence
        self.label = self.label_prob = None
        self.behavior = {}

        # structure
        self.prev = None
        self.next = None
        self.assign_id = None
        self.assign_label = None
        self.block_confidence = None

        # init process
        self._init_behavior()
    
    def _init_behavior(self):
        self.behavior = {
            'on_mouse': False
        }

    @property
    def area(self):
        return (self.pt2[0] - self.pt1[0] + 1) * (self.pt2[1] - self.pt1[1] + 1)
    
    @property
    def classification_label(self):
        return max(self._multiclass_result.items(), key=lambda x: x[1])

    @property
    def multiclass_result(self):
        return self._multiclass_result
    
    def l2_dist(self, bbox):
        center1 = np.array(self.center)
        center2 = np.array(bbox.center)
        return np.linalg.norm(center1 - center2)
    
    def calc_iou(self, bbox):
        xmin = max(self.pt1[0], bbox.pt1[0])
        ymin = max(self.pt1[1], bbox.pt1[1])
        xmax = min(self.pt2[0], bbox.pt2[0])
        ymax = min(self.pt2[1], bbox.pt2[1])

        if xmin < xmax and ymin < ymax:
            inter_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            iou = inter_area / float(self.area + bbox.area - inter_area + 1e-5)
        else:
            iou = 0

        return iou

class Frame(object):
    def __init__(self, frame_idx, bboxes):
        self.frame_idx = frame_idx
        self.bboxes = bboxes

        # 'fff' means forward from former
        self.bboxes_fff = False
    
    @property
    def bbox_count(self):
        return len(self.bboxes)

class TrackBlock(object):
    def __init__(self, *bboxes):
        self.bboxes = []

        self._label = None
        self._confidence = None
        self._frame_idx_set = set()
        self._nbboxes_true_label = None
        self._nbboxes_confidence = None
        self._nbboxes_frame_idx_set = None

        for b in bboxes:
            if not self.bboxes:
                self.bboxes = [b]
            else:
                self.append(b)

    @property
    def label(self):
        check_label_and_confidence = self.confidence
        return self._label

    @property
    def head(self):
        return self.bboxes[0]

    @property
    def tail(self):
        return self.bboxes[-1]
    
    @property
    def frame_idx_set(self):
        if self._nbboxes_frame_idx_set != len(self.bboxes):
            self._frame_idx_set = set(b.frame_idx for b in self.bboxes)
            self._nbboxes_frame_idx_set = len(self.bboxes)
        return self._frame_idx_set
    
    @property
    def confidence(self):
        if not self._nbboxes_confidence or self._nbboxes_confidence != len(self.bboxes):
            self.vote_for_label()
            bbox_labels = [b.multiclass_result[self._label] for b in self.bboxes]
            self._confidence = sum(bbox_labels) / len(bbox_labels)
            self._nbboxes_confidence = len(self.bboxes)
        return self._confidence

    @property
    def nbboxes_true_label(self):
        if not self._nbboxes_true_label:
            self.vote_for_label()
        return self._nbboxes_true_label
    
    def append(self, bbox):
        self.bboxes[-1].next = bbox
        bbox.prev = self.bboxes[-1]
        self.bboxes.append(bbox)

    def vote_for_label(self):
        labels = [b.classification_label[0] for b in self.bboxes]
        counter = Counter(labels)
        self._label, self._num_true_label_bboxes = counter.most_common(1)[0]
        return self._label
    
    def extract(self):
        node = self.head
        bboxes = [node]
        while node.next:
            bboxes.append(node.next)
            node = node.next
        self.bboxes = bboxes
        return bboxes

class TrackFlow(object):
    def __init__(self, reference_keys):
        self._ref_keys = reference_keys
        self._blocks = {k: [] for k in reference_keys}
        self._paths = {k: [] for k in reference_keys}
        self.logger = logging.getLogger(self.__class__.__name__)

        self.mouse_cnts = None
        self.mouse_skip_nframe = None
        self.check_on_mouse = False
        self.check_update_path = False
    
    @property
    def paths(self):
        if not self.check_update_path:
            self.flatten_block()
        return self._paths
    
    @property
    def blocks(self):
        return self._blocks

    def _block_padding(self, block1, block2):
        # make sure the block is further than block2
        if block1.bboxes[0].frame_idx > block2.bboxes[0].frame_idx:
            block1, block2 = block2, block1
        
        # merge
        score1 = block1.confidence * len(block1.bboxes)
        score2 = block2.confidence * len(block2.bboxes)
        bboxes = None
        if score1 >= score2:
            merge_frame_idx = block1.bboxes[-1].frame_idx
            bboxes = [b for b in block1.bboxes if b.frame_idx <= merge_frame_idx]
            bboxes += [b for b in block2.bboxes if b.frame_idx > merge_frame_idx]    
        else:
            merge_frame_idx = block2.bboxes[0].frame_idx
            bboxes = [b for b in block1.bboxes if b.frame_idx < merge_frame_idx]
            bboxes += [b for b in block2.bboxes if b.frame_idx >= merge_frame_idx]
        
        new_block = None
        for bid, bbox in enumerate(bboxes):
            if not new_block:
                new_block = TrackBlock(bbox)
            else:
                new_block.append(bbox)
        return new_block
    
    def _block_separating(self, block, frame_idx, break_on='right'):
        break_bbox = [b for b in block.bboxes if b.frame_idx >= frame_idx][0]
        break_bbox_index = block.bboxes.index(break_bbox)
        break_index = {'right': break_bbox_index, 'left': break_bbox_index+1}
        break_index = break_index.get(break_on, None)

        if break_index is None:
            self.logger.exception('Block seperating should be intersected or break on right or left')
            return

        l_block = TrackBlock(*block.bboxes[:break_index]) if break_index > 0 else None
        r_block = TrackBlock(*block.bboxes[break_index:]) if break_index < len(block.bboxes) else None
        return l_block, r_block

    def _block_merging(self, block1, block2):
        if not block1 and not block2:
            return None
        elif not block1 and block2:
            return block2
        elif not block2 and block1:
            return block1

        block_is_equal = block1.head.frame_idx == block2.head.frame_idx and \
                         block1.tail.frame_idx == block2.tail.frame_idx
        if block_is_equal:
            return block1 if block1.confidence > block2.confidence else block2
        elif block1.frame_idx_set & block2.frame_idx_set:
            intersect_frame_idx = sorted(list(block1.frame_idx_set & block2.frame_idx_set))
            block1_l, block1_m = self._block_separating(block1, intersect_frame_idx[0], 'right')
            block1_m, block1_r = self._block_separating(block1_m, intersect_frame_idx[-1], 'left')
            block2_l, block2_m = self._block_separating(block2, intersect_frame_idx[0], 'right')
            block2_m, block2_r = self._block_separating(block2_m, intersect_frame_idx[-1], 'left')
            block_l = self._block_merging(block1_l, block2_m)
            block_m = self._block_merging(block1_m, block2_m)
            block_r = self._block_merging(block1_r, block2_r)
            merge_block = []
            for block in [block_l, block_m, block_r]:
                if not block:
                    continue
                if isinstance(block, list):
                    merge_block += block
                else:
                    merge_block.append(block)
            return merge_block
        elif block1.label != block2.label:
            return [block1, block2]
        else:
            if block1.head.frame_idx > block2.head.frame_idx:
                block1, block2 = block2, block1
            if block1.tail.frame_idx + 1 == block2.head.frame_idx:
                if block1.tail.calc_iou(block2.head) > 0:
                    for bbox in block2.bboxes:
                        block1.append(bbox)
                    return block1
                else:
                    return block1 if block1.confidence > block2.confidence else block2

    def append_block(self, label, block):
        for bid, exist_block in enumerate(self._blocks[label]):
            # frame_idx intersection
            bbox_intersection = exist_block.frame_idx_set & block.frame_idx_set
            if bbox_intersection:

                # if conflict bbox overlap
                self._blocks[label].remove(exist_block)
                blocks = self._block_merging(exist_block, block)
                if isinstance(blocks, list):
                    self._blocks[label] += blocks
                else:
                    self._blocks[label].append(blocks)
                self.check_update_path = False
                return

        self._blocks[label].append(block)
        self.check_update_path = False

    def flatten_block(self):
        for label, blocks in self._blocks.items():
            self._paths[label] = []
            for block in blocks:
                self._paths[label] += block.bboxes
        self.check_update_path = True

class Mouse(object):
    def __init__(self):
        # attribute
        self.frame_idx = None
        self.contour = None
        self.contour_extend = None
        self.contour_extend_kernel = None
        self.center = None

    def _get_contour(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cnts
    
    def _get_modified_contour(self, shape, contour, mode):
        assert mode in ['dilate', 'erode'] and self.contour_extend_kernel is not None
        func_set = {'dilate': cv2.dilate, 'erode': cv2.erode}

        mask = np.zeros(shape)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        modified_mask = func_set[mode](mask, self.contour_extend_kernel, iterations=1)
        modified_mask = modified_mask.astype(np.uint8)[...]
        _, cnts, _ = cv2.findContours(modified_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cnts
    
    def _get_center(self, cnt):
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"]) if M["m00"] else 0
        cY = int(M["m01"] / M["m00"]) if M["m00"] else 0
        return np.array((cX, cY), dtype=np.int16)
        
    def _sort_contours_by_rect(self, cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
    
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
    
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
    
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
            key=lambda b:b[1][i], reverse=reverse))
    
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    def _point_in_bbox(self, point, bbox):
        x1, y1, x2, y2 = *bbox.pt1, *bbox.pt2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x, y = point
        return x1 <= x <= x2 and y1 <= y <= y2

    def update_by_frame_idx(self, frame_idx: int, frame, kernel_size):
        self.frame_idx = frame_idx
        self.contour = self._get_contour(frame)
        self.contour = sorted(self.contour, key=cv2.contourArea)[-1]
        self.contour_extend_kernel = np.ones(kernel_size)
        self.contour_extend = self._get_modified_contour(frame.shape[:2], self.contour, 'dilate')
        self.contour_extend = self.contour_extend[0]
        self.center = self._get_center(self.contour)

    def is_touch(self, bbox: BBox, frame_idx: int=None, frame=None):
        # update the mouse contour
        is_frame_idx_available = frame_idx is not None and frame_idx != self.frame_idx
        if is_frame_idx_available:
            assert frame is not None and frame_idx
            self.update_by_frame_idx(frame_idx , frame, self.contour_extend_kernel)
        
        # determine whether the bbox is touch mouse contour
        # checked == (1, 0 ,-1): (inside, on the edge, outside)
        ## bbox_rect_pts = (bbox.pt1, bbox.pt2, (bbox.pt1[0], bbox.pt2[1]), (bbox.pt2[0], bbox.pt1[1]))
        ## checked = map(lambda x: cv2.pointPolygonTest(self.contour, x, False), bbox_rect_pts)
        ## checked = any(i >= 0 for i in checked)
        checked = cv2.pointPolygonTest(self.contour_extend, bbox.center, False) >= 0
        return checked
