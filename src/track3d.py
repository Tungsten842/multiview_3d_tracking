import json
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class KalmanFilter3D:
    def __init__(self, init_pos, q_std=0.05, r_std=0.10):
        self.kf = cv2.KalmanFilter(6, 3, 0)

        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.transitionMatrix[0:3, 3:6] = np.eye(3, dtype=np.float32)

        self.kf.measurementMatrix = np.eye(3, 6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * (q_std**2)
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * (r_std**2)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.statePost = np.array([*init_pos, 0, 0, 0], dtype=np.float32).reshape(
            6, 1
        )

    def predict(self):
        return self.kf.predict()[:3].flatten()

    def update(self, measurement):
        meas_arr = measurement.reshape(3, 1)
        return self.kf.correct(meas_arr)[:3].flatten()


def load_camera_calibrations(
    calib_dir="camera",
    unit_scale=1.0 / 1000.0,
    orig_res=(3840, 2160),
    target_res=(1280, 704),
):
    files = [
        os.path.join(calib_dir, "4.json"),
        os.path.join(calib_dir, "13.json"),
    ]

    scale = target_res[0] / orig_res[0]

    # Calculate top crop offset
    scaled_height = orig_res[1] * scale
    crop_top = (scaled_height - target_res[1]) / 2.0

    cameras = []
    for fpath in files:
        with open(fpath, "r") as f:
            data = json.load(f)

        K = np.array(data["mtx"], dtype=np.float32)
        dist = np.array(data["dist"], dtype=np.float32)
        rvec = np.array(data["rvecs"], dtype=np.float32).reshape(3, 1)
        tvec = np.array(data["tvecs"], dtype=np.float32).reshape(3, 1) * unit_scale

        # Scale intrinsics for target resolution
        K[0, 0] *= scale
        K[0, 2] *= scale
        K[1, 1] *= scale
        K[1, 2] = K[1, 2] * scale - crop_top

        # Apply digital zoom scaling
        if os.path.basename(fpath) == "13.json":
            zoom = 1.20
            w, h = target_res
            K[0, 0] *= zoom
            K[1, 1] *= zoom
            K[0, 2] = (K[0, 2] - w / 2.0) * zoom + w / 2.0
            K[1, 2] = (K[1, 2] - h / 2.0) * zoom + h / 2.0

        R, _ = cv2.Rodrigues(rvec)
        center = (-R.T @ tvec).flatten()
        P_norm = np.hstack((R, tvec))

        cameras.append(
            {
                "K": K,
                "dist": dist,
                "R": R,
                "tvec": tvec,
                "center": center,
                "P_norm": P_norm,
            }
        )
    return cameras


class Tracker3D:
    def __init__(self, calib_dir, max_ray_dist=0.20, max_dist_3d=0.8, max_age=20):
        self.cameras = load_camera_calibrations(calib_dir)
        self.max_ray_dist = max_ray_dist
        self.max_dist_3d = max_dist_3d
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1

        # Precompute & cache static camera projection parameters
        cam0, cam1 = self.cameras[0], self.cameras[1]
        self.K0, self.dist0 = cam0["K"], cam0["dist"]
        self.K1, self.dist1 = cam1["K"], cam1["dist"]
        self.R0, self.t0 = cam0["R"], cam0["tvec"]
        self.R1, self.t1 = cam1["R"], cam1["tvec"]

        self.P0 = cam0["P_norm"]
        self.P1 = cam1["P_norm"]

    def _triangulate(self, multi_cam_tracks):
        t0, t1 = multi_cam_tracks[0], multi_cam_tracks[1]

        if len(t0) == 0 or len(t1) == 0:
            return [], []

        # Extract 2D centers and convert to normalized camera coordinates
        center0 = (t0[:, :2] + t0[:, 2:4]) * 0.5
        center1 = (t1[:, :2] + t1[:, 2:4]) * 0.5
        p0 = cv2.undistortPoints(center0, self.K0, self.dist0).squeeze(axis=1)
        p1 = cv2.undistortPoints(center1, self.K1, self.dist1).squeeze(axis=1)

        cls0, cls1 = t0[:, 6].astype(int), t1[:, 6].astype(int)

        # Find all pairs with same class id
        i0_m, i1_m = np.where(cls0[:, None] == cls1[None, :])
        if len(i0_m) == 0:
            return [], []

        # Triangulation
        pts4d = cv2.triangulatePoints(self.P0, self.P1, p0[i0_m].T, p1[i1_m].T)

        pts3d = (pts4d[:3] / pts4d[3]).T

        # Cheirality check and filter
        xc0 = pts3d @ self.R0.T + self.t0.ravel()
        xc1 = pts3d @ self.R1.T + self.t1.ravel()
        valid = (xc0[:, 2] > 0.05) & (xc1[:, 2] > 0.05)
        if not np.any(valid):
            return [], []
        i0_v, i1_v = i0_m[valid], i1_m[valid]
        pts3d_v, xc0_v, xc1_v = pts3d[valid], xc0[valid], xc1[valid]

        # Calculate ray distance
        err0 = np.linalg.norm(xc0_v[:, :2] - p0[i0_v] * xc0_v[:, 2:3], axis=1)
        err1 = np.linalg.norm(xc1_v[:, :2] - p1[i1_v] * xc1_v[:, 2:3], axis=1)
        ray_dist = (err0 + err1) * 0.5

        # Construct cost matrix
        cost_matrix = np.full((len(p0), len(p1)), 1e6, dtype=np.float32)
        pts_matrix = np.zeros((len(p0), len(p1), 3), dtype=np.float32)
        cost_matrix[i0_v, i1_v] = ray_dist
        pts_matrix[i0_v, i1_v] = pts3d_v

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Threshold filtering
        matched = cost_matrix[row_ind, col_ind] < self.max_ray_dist
        matched_rows, matched_cols = row_ind[matched], col_ind[matched]

        tri_pts = pts_matrix[matched_rows, matched_cols]
        tri_cls = cls0[matched_rows]

        return tri_pts, tri_cls

    def update(self, multi_cam_tracks):
        # Triangulate and predict
        tri_pts, tri_cls = self._triangulate(multi_cam_tracks)
        for trk in self.tracks.values():
            trk["age"] += 1
            trk["pos"] = trk["kf"].predict()

        matched_dets = set()

        # Match tracks to detections
        if self.tracks and len(tri_pts):
            track_ids = list(self.tracks.keys())
            preds = [self.tracks[i]["pos"] for i in track_ids]
            t_cls = [self.tracks[i]["class_id"] for i in track_ids]

            # Calculate distance matrix
            cost_matrix = cdist(preds, tri_pts)
            cost_matrix[np.array(t_cls)[:, None] != np.array(tri_cls)[None, :]] = 1e6

            # Hungarian algorithm
            rows, cols = linear_sum_assignment(cost_matrix)

            valid = cost_matrix[rows, cols] < self.max_dist_3d
            for r, c in zip(rows[valid], cols[valid]):
                trk = self.tracks[track_ids[r]]
                trk["pos"] = trk["kf"].update(tri_pts[c])
                trk["age"] = 0
                trk["hits"] += 1
                matched_dets.add(c)

        # Add unmatched detections as new tracks
        for c in set(range(len(tri_pts))) - matched_dets:
            self.tracks[self.next_id] = {
                "kf": KalmanFilter3D(tri_pts[c]),
                "pos": tri_pts[c],
                "class_id": tri_cls[c],
                "age": 0,
                "hits": 1,
            }
            self.next_id += 1

        # Remove dead tracks
        self.tracks = {k: v for k, v in self.tracks.items() if v["age"] <= self.max_age}

        return [
            {"id": k, "pos": v["pos"], "class_id": v["class_id"]}
            for k, v in self.tracks.items()
            if v["hits"] >= 2 or v["age"] == 0
        ]
