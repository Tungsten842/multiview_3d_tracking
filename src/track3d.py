import json
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class KalmanFilter3D:
    def __init__(
        self,
        init_pos,
        dt=1.0 / 25.0,
        process_std=2.0,
        measurement_std=0.10,
        v_max=8.0,
        is_ball=False,
    ):
        self.dt = dt
        self.is_ball = is_ball
        self.kf = cv2.KalmanFilter(6, 3, 0)

        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.transitionMatrix[0:3, 3:6] = np.eye(3, dtype=np.float32) * dt

        self.kf.measurementMatrix = np.eye(3, 6, dtype=np.float32)

        # Process noise covariance (Piecewise Constant White Acceleration model)
        q_1d = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]], dtype=np.float32)
        self.kf.processNoiseCov = np.kron(q_1d, np.eye(3, dtype=np.float32)) * (
            process_std**2
        )

        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * (measurement_std**2)

        # Set position and velocity uncertainty
        v_std = v_max / 3.0
        p_diag = [measurement_std**2] * 3 + [v_std**2] * 3
        self.kf.errorCovPost = np.diag(p_diag).astype(np.float32)

        self.kf.statePost = np.array([*init_pos, 0, 0, 0], dtype=np.float32).reshape(
            6, 1
        )

    def predict(self, missed=False, damping_factor=0.96):
        alpha = damping_factor if missed else 1.0
        # Set velocity decay
        self.kf.transitionMatrix[3:6, 3:6] = np.eye(3, dtype=np.float32) * alpha
        return self.kf.predict()[:3].flatten()


    def update(self, measurement):
        meas_arr = measurement.reshape(3, 1)
        predicted_pos = self.kf.statePre[:3]
        dist = np.linalg.norm(meas_arr - predicted_pos)

        # Handle sudden direction changes
        if self.is_ball and dist > 1.2:
            self.kf.errorCovPre[3:6, 3:6] += np.eye(3, dtype=np.float32) * 10.0

        return self.kf.correct(meas_arr)[:3].flatten()


def load_camera_calibrations(
    calib_dir="camera",
    unit_scale=1.0 / 1000.0,
    orig_res=(3840, 2160),
    target_res=(1280, 704),
):
    files = [
        os.path.join(calib_dir, "2.json"),
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
    def __init__(
        self, calib_dir, max_ray_dist=0.25, max_dist_3d=1.0, max_age=25, min_hits=2
    ):
        self.cameras = load_camera_calibrations(calib_dir)
        self.max_ray_dist = max_ray_dist
        self.max_dist_3d = max_dist_3d
        self.max_age = max_age
        self.min_hits = min_hits
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
            return [], [], []

        # Extract 2D centers and convert to normalized camera coordinates
        center0 = (t0[:, :2] + t0[:, 2:4]) * 0.5
        center1 = (t1[:, :2] + t1[:, 2:4]) * 0.5
        p0 = cv2.undistortPoints(center0, self.K0, self.dist0).squeeze(axis=1)
        p1 = cv2.undistortPoints(center1, self.K1, self.dist1).squeeze(axis=1)

        cls0, cls1 = t0[:, 6].astype(int), t1[:, 6].astype(int)

        # Find all pairs with same class id
        i0_m, i1_m = np.where(cls0[:, None] == cls1[None, :])
        if len(i0_m) == 0:
            return [], [], []

        # Triangulation
        pts4d = cv2.triangulatePoints(self.P0, self.P1, p0[i0_m].T, p1[i1_m].T)

        pts3d = (pts4d[:3] / pts4d[3]).T

        # Cheirality check and filter
        xc0 = pts3d @ self.R0.T + self.t0.ravel()
        xc1 = pts3d @ self.R1.T + self.t1.ravel()
        valid = (xc0[:, 2] > 0.05) & (xc1[:, 2] > 0.05)
        if not np.any(valid):
            return [], [], []

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
        tri_ids = np.column_stack((t0[matched_rows, 4], t1[matched_cols, 4]))

        return tri_pts, tri_cls, tri_ids

    def update(self, multi_cam_tracks):
        # Triangulate and predict
        tri_pts, tri_cls, tri_ids = self._triangulate(multi_cam_tracks)
        for trk in self.tracks.values():
            trk["age"] += 1
            is_missed = trk["age"] > 1
            trk["pos"] = trk["kf"].predict(missed=is_missed)

        matched_dets = set()

        # Match tracks to detections
        if self.tracks and len(tri_pts):
            track_ids = list(self.tracks.keys())
            preds = [self.tracks[i]["pos"] for i in track_ids]
            t_cls = [self.tracks[i]["class_id"] for i in track_ids]

            # Calculate distance matrix
            cost_matrix = cdist(preds, tri_pts)
            cost_matrix[np.array(t_cls)[:, None] != np.array(tri_cls)[None, :]] = 1e4

            # Discount cost if 2D ByteTrack IDs match
            for r, tid in enumerate(track_ids):
                prev_ids = self.tracks[tid].get("cam_ids")
                if prev_ids is not None:
                    matches = (tri_ids[:, 0] == prev_ids[0]) | (
                        tri_ids[:, 1] == prev_ids[1]
                    )
                    cost_matrix[r, matches] *= 0.3

            # Hungarian algorithm
            rows, cols = linear_sum_assignment(cost_matrix)
            for r, c in zip(rows, cols):
                max_dist = 4.0 if tri_cls[c] == 1 else self.max_dist_3d
                if cost_matrix[r, c] >= max_dist:
                    continue
                trk = self.tracks[track_ids[r]]
                trk["pos"] = trk["kf"].update(tri_pts[c])
                trk["age"] = 0
                trk["cam_ids"] = tri_ids[c]

                trk["hits"] += 1
                matched_dets.add(c)

        # Add unmatched detections as new tracks
        for c in set(range(len(tri_pts))) - matched_dets:
            is_ball = tri_cls[c] == 1
            kf = (
                KalmanFilter3D(
                    tri_pts[c],
                    process_std=25.0,
                    measurement_std=0.05,
                    v_max=35.0,
                    is_ball=True,
                )
                if is_ball
                else KalmanFilter3D(tri_pts[c])
            )
            self.tracks[self.next_id] = {
                "kf": kf,
                "pos": tri_pts[c],
                "class_id": tri_cls[c],
                "cam_ids": tri_ids[c],
                "age": 0,
                "hits": 1,
            }
            self.next_id += 1

        # Remove dead tracks
        self.tracks = {k: v for k, v in self.tracks.items() if v["age"] <= self.max_age}

        return [
            {"id": k, "pos": v["pos"], "class_id": v["class_id"]}
            for k, v in self.tracks.items()
            if v["hits"] >= self.min_hits and v["age"] <= 15
        ]
