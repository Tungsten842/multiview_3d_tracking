import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker, DIoU
from trackers.utils.state_representations import XCYCSRStateEstimator


class Tracker2D:
    def __init__(self, num_cams, conf_threshold, nms_threshold, ball_class_id=1):
        self.num_cams = num_cams
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.ball_class_id = ball_class_id
        self.trackers = [
            ByteTrackTracker(frame_rate=25, iou=DIoU()) for _ in range(num_cams)
        ]
        self.ball_trackers = [
            ByteTrackTracker(
                frame_rate=25,
                state_estimator_class=XCYCSRStateEstimator,
                track_activation_threshold=0.2,
                iou=DIoU(),
            )
            for _ in range(num_cams)
        ]

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

                nms_indices = cv2.dnn.NMSBoxes(
                    boxes_xywh,
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

            sv_detections = sv.Detections(
                xyxy=detections[:, :4].astype(np.float32),
                confidence=detections[:, 4].astype(np.float32),
                class_id=detections[:, 5].astype(int),
            )

            # Split detections by class
            is_ball = sv_detections.class_id == self.ball_class_id
            ball_dets = sv_detections[is_ball]
            other_dets = sv_detections[~is_ball]

            # Track independently
            other_tracks = self.trackers[i].update(other_dets)
            ball_tracks = self.ball_trackers[i].update(ball_dets)

            # Merge results
            merged = sv.Detections.merge(
                [t for t in (other_tracks, ball_tracks) if len(t) > 0]
            )
            tracks = (
                np.column_stack(
                    (
                        merged.xyxy,
                        merged.tracker_id,
                        merged.confidence,
                        merged.class_id,
                    )
                ).astype(np.float32)
                if len(merged) > 0
                else np.empty((0, 7), dtype=np.float32)
            )
            # Discart unconfirmed tracks
            tracks = tracks[tracks[:, 4] > 0]
            results.append(tracks)

        return results
