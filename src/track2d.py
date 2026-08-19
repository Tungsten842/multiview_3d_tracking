import cv2
import numpy as np
from boxmot.trackers.bbox.bytetrack import ByteTrack


class Tracker2D:
    def __init__(
        self,
        num_cams,
        conf_threshold,
        nms_threshold,
    ):
        self.num_cams = num_cams
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.trackers = [ByteTrack(frame_rate=25) for _ in range(num_cams)]
        self.img = np.empty((1, 1))

    def update(self, predictions, frames):
        results = []

        for i in range(self.num_cams):
            pred = predictions[i].T
            boxes, class_scores = pred[:, :4], pred[:, 4:]

            scores = np.max(class_scores, axis=1)
            cls_ids = np.argmax(class_scores, axis=1)

            mask = scores > self.conf_threshold
            boxes, scores, cls_ids = boxes[mask], scores[mask], cls_ids[mask]

            detections = np.empty((0, 6), dtype=np.float32)

            if len(boxes) > 0:
                boxes_xywh = boxes.copy()
                boxes_xywh[:, :2] -= boxes_xywh[:, 2:] / 2.0

                # Shift coordinates by class ID so different classes never overlap
                max_dim = 10000.0
                boxes_offset = boxes_xywh.copy()
                boxes_offset[:, :2] += cls_ids[:, None] * max_dim

                nms_indices = cv2.dnn.NMSBoxes(
                    boxes_offset,
                    scores,
                    0.0,
                    self.nms_threshold,
                )

                if len(nms_indices) > 0:
                    idx = np.asarray(nms_indices).flatten()

                    # Convert original un-shifted boxes from xywh -> xyxy
                    boxes_xyxy = boxes_xywh[idx].copy()
                    boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

                    detections = np.column_stack(
                        (boxes_xyxy, scores[idx], cls_ids[idx])
                    )

            tracks = self.trackers[i].update(detections, self.img)
            results.append(tracks)

        return results
