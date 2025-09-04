import os
import sys
import cv2
import math
import argparse
import time
import numpy as np 
from collections import deque

# Ensure the repo root is on PYTHONPATH
REPO = os.path.abspath(os.path.join(__file__, "AVTrack"))
if REPO not in sys.path:
    sys.path.append(REPO)

from lib.test.evaluation.tracker import Tracker as TrackerWrapper
from util import RealSenseCamera, PointCloudVisualizer, BranchCurveFitter, PoseSmoother
from poser import BranchPoser

class LiveTracker:
    def __init__(self, source=0, tracker_name="avtrack", tracker_param="deit_tiny_patch16_224", draw=False, show_stat=False, video_path=None, output_path=None, visualize_pc=False):
        self.source = source
        self.tracker_name = tracker_name
        self.tracker_param = tracker_param
        self.draw = draw
        self.show_stat = show_stat
        self.video_path = video_path
        self.output_path = output_path
        self.visualize_pc = visualize_pc
        self.smoother = PoseSmoother(pos_alpha=0.40, dir_alpha=0.1, max_rot_deg=20.0, keep_sign=True)

        self.cap = None
        self.tracker = None
        self.writer = None
        self.frame_count = 0
        self.start_time = None
        self.last_report_time = None
        self.fps_display = 0.0
        self.frame_size = None
        self.fit = None
        self.pose = None 
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.bboxes = [] # store (frame_idx, x, y, w, h)

        # ----- simple/intense smoothing knobs -----
        self.smooth_enabled   = True
        self.smooth_win       = 29          # median window (odd; try 7, 9, 11)
        self.ema_alpha_pos    = 0.20       # 0.1–0.3 -> heavier smoothing
        self.ema_alpha_dir    = 0.20

        # buffers & state
        self._buf_P = deque(maxlen=self.smooth_win)
        self._buf_A = deque(maxlen=self.smooth_win)
        self._ema_P = None
        self._ema_A = None
        self._prev_T = None                # for tangent sign consistency

    def setup_tracker(self):
        """
        Build the tracker wrapper and instantiate the actual tracker.
        """
        wrapper = TrackerWrapper(self.tracker_name, self.tracker_param, dataset_name=None)
        params = wrapper.get_parameters()
        setattr(params, 'debug', False)
        self.tracker = wrapper.create_tracker(params)

    def open_source(self):
        """
        Open video capture (camera or file).
        """
        if isinstance(self.source, str):
            self.cap = cv2.VideoCapture(self.source)
            _, frame = self.cap.read()
        else:
            self.cap = RealSenseCamera(width=640, height=480, fps=60)
            frame, _ = self.cap.get_frames()
            self.poser = BranchPoser(self.cap, debug=False)
            if self.visualize_pc:
                self.visualizer = PointCloudVisualizer(self.cap)
                self.visualizer.cloud_gray = 0.35     # mute cloud; set None to keep texture
                self.visualizer.voxel_size = 0.01     # downsample ~1 cm (optional)
                self.visualizer.curve_radius = 0.002  # thicker tube
        # prepare video writer if requested
        if self.video_path:
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            self.writer = cv2.VideoWriter(self.video_path, self.fourcc, 30.0, self.frame_size)
        return frame

    def select_initial_bbox(self, frame):
        """
        Let user select ROI on the first frame.
        This will be replaced by an external detector later.
        """
        bbox = cv2.selectROI("Select branch", frame, showCrosshair=False, fromCenter=False)
        cv2.destroyWindow("Select branch")
        return list(map(int, bbox))

    def initialize(self, frame, bbox):
        """
        Initialize the tracker with the first frame and bounding box.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.tracker.initialize(rgb, {'init_bbox': bbox})

    def update_fps(self):
        """
        Compute and update FPS once per second.
        """
        current_time = time.time()
        self.frame_count += 1
        elapsed_since_report = current_time - self.last_report_time
        total_elapsed = current_time - self.start_time
        if elapsed_since_report >= 1.0:
            self.fps_display = self.frame_count / total_elapsed
            self.last_report_time = current_time
    
    def compute_sample_and_pose(self, bbox, prefer_front=True, smooth=True):
        """
        Choose a 3D sample point on the fitted curve using the camera ray through bbox center.
        Returns dict with: idx, point (3,), tangent (3,), approach (3,), pose (4x4).
        """
        if self.fit is None or 'curve' not in self.fit or 'tangent' not in self.fit:
            return None
        curve = np.asarray(self.fit['curve'], dtype=np.float32)
        tang  = np.asarray(self.fit['tangent'], dtype=np.float32)
        if len(curve) < 2 or len(tang) != len(curve):
            return None
        if not isinstance(self.cap, RealSenseCamera):
            return None

        # camera ray through bbox center
        intr = self.cap.get_intrinsics()['color']
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        x, y, w, h = map(int, bbox)
        u = x + 0.5 * w
        v = y + 0.5 * h
        d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float32)
        d /= (np.linalg.norm(d) + 1e-9) # unit direciton from the camera origin to the sampling point

        # distance from points to ray through origin with direction d:
        # dist^2 = ||P - (P·d) d||^2
        P = curve
        proj = (P @ d)[:, None] * d
        off  = P - proj
        dist2 = np.sum(off * off, axis=1)

        # optionally require points "in front" of camera (P·d > 0)
        if prefer_front:
            front = (P @ d) > 0
            if np.any(front):
                idx = int(np.argmin(np.where(front, dist2, np.inf)))
            else:
                idx = int(np.argmin(dist2))
        else:
            idx = int(np.argmin(dist2))

        p_sel = P[idx]
        t_sel = tang[idx]
        t_sel = t_sel / (np.linalg.norm(t_sel) + 1e-9)

        # approach = component of camera->point vector orthogonal to tangent
        v_cam = p_sel / (np.linalg.norm(p_sel) + 1e-9) # unit vector from camera origin to the sample point
        approach = v_cam - np.dot(v_cam, t_sel) * t_sel
        nrm = np.linalg.norm(approach)
        if nrm < 1e-6:
            approach = v_cam
        else:
            approach /= nrm

        # temporal smoothing 
        if smooth:
            p_sel, approach = self.smoother.update(p_sel, approach)
        # Build pose with smoothed values
        z = approach
        yv = t_sel - np.dot(t_sel, z) * z
        yv /= (np.linalg.norm(yv) + 1e-9)
        xv = np.cross(yv, z); xv /= (np.linalg.norm(xv) + 1e-9)
        yv = np.cross(z, xv); yv /= (np.linalg.norm(yv) + 1e-9)

        T = np.eye(4, dtype=np.float32)
        T[:3, 0] = xv
        T[:3, 1] = yv
        T[:3, 2] = z
        T[:3, 3] = p_sel.astype(np.float32)

        return {'idx': idx, 'point': p_sel, 'tangent': t_sel, 'approach': approach, 'pose': T}

    def draw_overlay(self, frame, bbox):
        """
        Draw tracking box and FPS overlay on the frame.
        """
        x, y, w, h = bbox
        # bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # FPS
        if self.show_stat:
            cv2.putText(frame, f"FPS: {self.fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # Curve overlay (only if we have a fit AND a RealSense camera)
        if self.fit is not None and isinstance(self.cap, RealSenseCamera):
            H, W = frame.shape[:2]

            # project smooth curve
            if 'curve' in self.fit and self.fit['curve'] is not None and len(self.fit['curve']) >= 2:
                uv, valid = self._project_to_image(self.fit['curve'])
                if uv is not None:
                    # clip to frame
                    inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                    good = valid & inside
                    # draw polyline as segments between consecutive valid points
                    for i in range(1, len(uv)):
                        if good[i - 1] and good[i]:
                            cv2.line(frame, tuple(uv[i - 1]), tuple(uv[i]), (0, 255, 255), 2) # YELLOW branch

            # sampling pose (contact point + approach)
            if self.pose:
                P = self.pose['point']        # (3,)
                A = self.pose['approach']     # (3,) unit
                T = self.pose['tangent']      # (3,) unit

                # project the selected 3D point
                uv_pt, ok_pt = self._project_to_image(P[None, :])
                if uv_pt is not None and ok_pt[0]:
                    u, v = int(uv_pt[0, 0]), int(uv_pt[0, 1])
                    if 0 <= u < W and 0 <= v < H:
                        # draw contact point
                        cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)  # red dot

                        # draw short approach arrow (e.g., 5 cm)
                        tip3d = P + 0.05 * A
                        uv_tip, ok_tip = self._project_to_image(tip3d[None, :])
                        if uv_tip is not None and ok_tip[0]:
                            uu, vv = int(uv_tip[0, 0]), int(uv_tip[0, 1])
                            if 0 <= uu < W and 0 <= vv < H:
                                cv2.arrowedLine(frame, (u, v), (uu, vv), (0, 0, 255), 2, tipLength=0.3)


    def run(self):
        """
        Main loop: open source, select ROI, initialize tracker, process frames.
        """
        # Setup
        self.setup_tracker()
        first_frame = self.open_source()
        bbox = self.select_initial_bbox(first_frame)
        self.initialize(first_frame, bbox)

        # FPS timers
        self.start_time = time.time()
        self.last_report_time = self.start_time

        try:
            while True:
                if isinstance(self.source, str):
                    _, frame = self.cap.read() 
                else:
                    frame, depth = self.cap.get_frames()

                # Update FPS counter
                if self.show_stat:
                    self.update_fps()

                # Track
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out = self.tracker.track(rgb)
                if isinstance(out, dict):
                    target = list(map(int, out['target_bbox']))
                else:
                    target = list(map(int, out))
                # Find the pointcloud given bbox
                self.poser.segment_roi(target, frame, depth) # will update self.mask_roi if mask is valid
                roi = self.poser.mask_roi
                if roi:
                    pc = self.poser.refine_pointcloud(target, max_points=2000, voxel=0.01)
                    verts, tex = pc['verts'], pc['tex']
                    # at current view, the mask is valid
                    if len(verts) > 2:
                        verts, tex = self.poser.nearest_cluster(verts, tex, method='dbscan', eps=0.025, min_points=5)
                        cf = BranchCurveFitter(nbins=12, min_per_bin=18, ema_ctrl=0.1, ema_axis=0.1, ema_span=0.1)
                        self.fit = cf.fit_stiff(verts, deg=4)
                        #self.fit = cf.fit(verts) # use when the branch has weird shape
                        # sample the pose only if current mask is solid
                        self.pose = self.compute_sample_and_pose(target, smooth=True)
                    else: 
                        self.fit = None 
                        self.pose = None 
                    # for DEBUG purpose
                    if 'overlay' in roi:
                        cv2.imshow('Branch FG (Depth+GrabCut)', roi['overlay'])
                else: 
                    verts, tex = self.cap.get_pointcloud(target)
                
                # for DEBUG pointcloud
                if self.visualize_pc:
                    if self.fit:
                        self.visualizer.set_curve(self.fit['curve'], color = (1,1,0))
                    if self.pose:
                        self.visualizer.set_sample(self.pose['point'], self.pose['approach'], point_color=(1,0,0), arrow_color=(1,0,0))
                    self.visualizer.update(verts, tex)

                # output the bbox if needed
                if self.output_path:
                    self.bboxes.append((self.frame_count, *target))
                if self.draw:
                    # Draw overlay and display
                    self.draw_overlay(frame, target)
                    cv2.imshow("AVTrack Online", frame)

                # write frame if writer enabled
                if self.writer:
                    self.writer.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Release resources and print average FPS.
        """
        # save bounding boxes if needed
        if self.output_path:
            with open(self.output_path, 'w') as f:
                f.write('frame, x, y, w, h\n')
                for fr, x, y, w, h in self.bboxes:
                    f.write(f"{fr}, {x}, {y}, {w}, {h}\n")
                print(f"Saved {len(self.bboxes)} bboxes to {self.output_path}")
        if self.show_stat:
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            print(f"Tracked {self.frame_count} frames in {total_time:.2f} seconds."
                  f" Average FPS: {avg_fps:.2f}")
        if self.cap:
            try:
                self.cap.stop() 
            except:
                self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()


    # Helper Functions 
    def _project_to_image(self, pts3d):
        """
        Project Nx3 camera-frame points (meters) to pixel coords using RealSense intrinsics.
        Returns (uv_int Nx2 int32, valid_mask Nx bool).
        """
        if not isinstance(self.cap, RealSenseCamera):
            return None, None
        intr = self.cap.get_intrinsics()['color']
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

        pts = np.asarray(pts3d, dtype=np.float32)
        Z = pts[:, 2]
        valid = Z > 1e-6
        U = fx * (pts[:, 0] / np.maximum(Z, 1e-6)) + cx
        V = fy * (pts[:, 1] / np.maximum(Z, 1e-6)) + cy
        uv = np.stack([U, V], axis=1).astype(np.int32)
        return uv, valid
    
    def _median_vec(self, vecs):
        """Component-wise median of a list of 3D vectors."""
        arr = np.asarray(vecs, dtype=np.float32)
        return np.median(arr, axis=0)

    def _simple_smooth(self, P, A, T):
        """
        Heavy smoothing for sample point P and approach A:
        - keep tangent T sign-consistent
        - sliding median over last K frames
        - exponential moving average (EMA)
        - project direction into plane ⟂ T and renormalize
        Returns (P_sm, A_sm).
        """
        if not self.smooth_enabled or P is None or A is None:
            return P, A

        # 0) tangent sign consistency (avoid flips)
        Tn = T / (np.linalg.norm(T) + 1e-9)
        if self._prev_T is not None and np.dot(Tn, self._prev_T) < 0:
            Tn = -Tn
        self._prev_T = Tn.copy()

        # 1) push into buffers
        self._buf_P.append(P)
        # keep direction sign consistent with previous EMA to avoid 180° flips
        A_use = A.copy()
        if self._ema_A is not None and np.dot(A_use, self._ema_A) < 0:
            A_use = -A_use
        self._buf_A.append(A_use)

        # 2) sliding median
        P_med = self._median_vec(self._buf_P)
        A_med = self._median_vec(self._buf_A)

        # 3) EMA
        if self._ema_P is None:
            self._ema_P = P_med.copy()
            self._ema_A = A_med.copy()
        else:
            self._ema_P = (1.0 - self.ema_alpha_pos) * self._ema_P + self.ema_alpha_pos * P_med
            self._ema_A = (1.0 - self.ema_alpha_dir) * self._ema_A + self.ema_alpha_dir * A_med

        # 4) project approach into plane ⟂ T and normalize
        A_proj = self._ema_A - np.dot(self._ema_A, Tn) * Tn
        n = np.linalg.norm(A_proj)
        if n < 1e-6:
            A_sm = A_use / (np.linalg.norm(A_use) + 1e-9)  # fallback
        else:
            A_sm = A_proj / n

        return self._ema_P.copy(), A_sm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run AVTrack online with live FPS display and optional recording'
    )
    parser.add_argument('--tracker_name', type=str, default='avtrack',
                        help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='deit_tiny_patch16_224',
                        help='Name of config file.')
    parser.add_argument('--source', type=str, default=None,
                        help='Camera index or video file path.')
    parser.add_argument('--draw_overlay', action='store_true',
                        help='draw bbox, curve, etc')
    parser.add_argument('--show_stat', action='store_true',
                        help='Show fps counter')
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to save output video')
    parser.add_argument('--output_path', type=str, default=None, 
                        help='Path to save output bounding boxes')
    parser.add_argument('--visualize_pc', action='store_true', 
                        help='Whether visualize the pointcloud constructed by Realsense, NOTE it assumes the source to be a Realsense camera')
    args = parser.parse_args()

    # Convert source to int for camera if digit
    source = args.source
    live = LiveTracker(source, args.tracker_name, args.tracker_param, args.draw_overlay, args.show_stat, args.video_path, args.output_path, args.visualize_pc)
    live.run()
