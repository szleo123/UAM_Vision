# poser.py
import numpy as np
import cv2
import open3d as o3d

class BranchPoser:
    """
    Minimal branch foreground segmenter (no training).
    Assumes RealSense frames are aligned to COLOR (align_to='color').
    """

    def __init__(self, camera,
                 exg_thresh=20,           # Excess-G threshold
                 bark_sat_max=60,         # low-saturation -> bark/wood
                 depth_band_min=0.02,     # >= 2 cm band around depth mode
                 depth_band_frac=0.12,    # or 12% of depth
                 gc_iters=2,              # GrabCut iterations
                 debug=False):
        self.cam = camera
        self.exg_thresh   = int(exg_thresh)
        self.bark_sat_max = int(bark_sat_max)
        self.depth_band_min  = float(depth_band_min)
        self.depth_band_frac = float(depth_band_frac)
        self.gc_iters = int(gc_iters)
        self.debug = debug
        self.mask_roi = None # only get updated if the mask have value at some point
        self.depth_scale = self.cam.get_intrinsics()['depth_scale']

    def segment_roi(self, bbox, color_image, depth_image, return_overlay=False):
        """
        Segment foreground (branch) inside bbox using depth + color + GrabCut.

        Args:
            bbox: (x, y, w, h) in COLOR pixel coords.
            color_image: optional BGR frame (to avoid extra grab).
            depth_image: optional uint16 depth (same frame).
            return_overlay: if True (or debug=True), also returns 'overlay' image.

        Returns dict:
            {
              'mask_full': HxW uint8 (0/255),
              'mask_roi' : h x w uint8 (0/255),
              'bbox'     : (x, y, w, h),
              'overlay'  : BGR (only if requested/debug)
            }
            or None if frames unavailable.
        """

        H, W = color_image.shape[:2]
        x, y, w, h = map(int, bbox)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            return {
                'mask_full': np.zeros((H, W), np.uint8),
                'mask_roi':  np.zeros((0, 0), np.uint8),
                'bbox': (x1, y1, max(0, x2 - x1), max(0, y2 - y1))
            }

        # ROI views
        color = color_image[y1:y2, x1:x2]
        depth = depth_image[y1:y2, x1:x2].astype(np.float32) * self.depth_scale

        # 1) Depth seed: keep pixels near the nearest depth mode
        valid = depth > 0
        if np.count_nonzero(valid) > 50:
            dvals = depth[valid]
            q5, q95 = np.percentile(dvals, [5, 95])
            span = max(0.02, float(q95 - q5))
            bins = max(32, min(128, int(span / 0.01)))
            hist, edges = np.histogram(dvals, bins=bins, range=(q5, q95))
            center = (np.median(dvals) if hist.sum() == 0
                      else 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]))
            band = max(self.depth_band_min, self.depth_band_frac * center)
            depth_fg = (np.abs(depth - center) <= band).astype(np.uint8)
        else:
            depth_fg = (depth > 0).astype(np.uint8)

        # 2) Seed GrabCut with depth
        probable_fg = depth_fg.astype(np.uint8)
        gc_mask = np.full(color.shape[:2], cv2.GC_PR_BGD, np.uint8)  # 2
        gc_mask[probable_fg.astype(bool)] = cv2.GC_PR_FGD               # 3
        ring = 2  # border as sure background
        gc_mask[:ring, :] = cv2.GC_BGD; gc_mask[-ring:, :] = cv2.GC_BGD
        gc_mask[:, :ring] = cv2.GC_BGD; gc_mask[:, -ring:] = cv2.GC_BGD

        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(color, gc_mask, None, bgd, fgd,
                        iterCount=self.gc_iters, mode=cv2.GC_INIT_WITH_MASK)
            mask_roi = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                                255, 0).astype(np.uint8)
        except cv2.error:
            # Fallback if GrabCut fails
            mask_roi = (probable_fg > 0).astype(np.uint8) * 255

        # Clean up mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, k, iterations=1)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, k, iterations=1)

        # Stitch back to full image
        mask_full = np.zeros((H, W), np.uint8)
        mask_full[y1:y2, x1:x2] = depth_fg

        out = {
            'mask_full': mask_full,
            'mask_roi': mask_roi,
            'bbox': (x1, y1, x2 - x1, y2 - y1),
        }
        if return_overlay or self.debug:
            overlay = color_image.copy()
            roi = overlay[y1:y2, x1:x2]
            bg = roi.copy()
            # dim background, keep FG bright
            bg[mask_roi == 0] = (bg[mask_roi == 0] * 0.3).astype(np.uint8)
            overlay[y1:y2, x1:x2] = bg
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            out['overlay'] = overlay
        
        #if np.sum(mask_full) > 0: self.mask_roi = out
        self.mask_roi = out
        return out

    def refine_pointcloud(self,
                          bbox,
                          max_points=None, 
                          grid=32,
                          voxel=None,
                          seed=None):
        """
        Filter ROI point cloud using a binary mask.

        Args:
            bbox: (x,y,w,h) on COLOR image.
            mask_full: full-image mask (0/255). If provided, used directly.
            mask_roi: ROI-only mask (0/255). Used if mask_full is None.
            color_image: optional BGR image for sampling colors.
            return_colors: if True and color_image is provided, returns Nx3 float colors in [0,1].

        Returns:
            dict with keys:
              'verts'      : (N,3) float32 camera-frame points
              'tex'        : (N,2) float32 normalized texture coords
              'colors'     : (N,3) float32 in [0,1] (only if requested)
              'num_points' : int
              'keep_ratio' : float in [0,1]
        """
        # Get ROI-only point cloud quickly (your util.py does ROI back-projection when bbox is given)
        verts, tex = self.cam.get_pointcloud(bbox=bbox, max_points=max_points, voxel=voxel)
        if verts is None or len(verts) == 0:
            return {'verts': np.zeros((0,3), np.float32),
                    'tex':   np.zeros((0,2), np.float32),
                    'num_points': 0, 'keep_ratio': 0.0}

        # Determine image size for (u,v) pixels
        H, W = self.cam.height, self.cam.width

        # Map normalized tex -> pixel indices
        u = np.clip((tex[:, 0] * W).astype(np.int32), 0, W - 1)
        v = np.clip((tex[:, 1] * H).astype(np.int32), 0, H - 1)
        keep = np.ones(len(u), dtype=bool)

        # Filter
        verts_f = verts[keep].astype(np.float32)
        tex_f   = tex[keep].astype(np.float32)

        out = {
            'verts': verts_f,
            'tex':   tex_f,
            'num_points': int(verts_f.shape[0]),
            'keep_ratio': float(np.mean(keep)) if len(keep) else 0.0
        }

        return out

    def nearest_cluster(self,
                    verts: np.ndarray,
                    tex: np.ndarray,
                    method: str = "dbscan",
                    eps: float = None,
                    min_points: int = 10):
        """
        Keep the point-cloud cluster that is nearest to the camera.

        Args:
            verts: (N,3) float32 camera-frame points (meters).
            tex:   (N,2) float32 normalized texture coords.
            method: currently only 'dbscan' supported; others fall back to input.
            eps: DBSCAN radius (meters). If None, auto from NND if available.
            min_points: DBSCAN min_points.

        Returns:
            (verts_f, tex_f) or (verts_f, tex_f, keep_mask)
        """
        N = 0 if verts is None else len(verts)
        if N == 0:
            empty = (np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32))
            return empty

        # default fallbacks in case clustering fails
        keep_mask = np.ones(N, dtype=bool)   # fallback: keep all
        verts_f = verts
        tex_f = tex

        use_dbscan = (method in ("dbscan", "auto")) and ("o3d" in globals()) and (o3d is not None)
        if use_dbscan and N >= max(min_points, 20):
            try:
                if eps is None:
                    # auto eps via median nearest-neighbor distance in 3D
                    # (small, robust heuristic; tune if needed)
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(verts)
                    dists, _ = nn.kneighbors(verts)
                    med_nnd = float(np.median(dists[:, 1]))
                    eps = max(0.5 * med_nnd, 0.005)  # clamp to >=5mm
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(verts.astype(np.float32))
                labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

                if labels.size and labels.max() >= 0:
                    # choose cluster with smallest median positive Z (nearest to camera)
                    Z = verts[:, 2].astype(np.float32)
                    clusters = np.unique(labels[labels >= 0])

                    best_label, best_depth = None, None
                    for lab in clusters:
                        m = (labels == lab)
                        if np.count_nonzero(m) < min_points:
                            continue
                        zc = Z[m]
                        z_front = zc[zc > 0]
                        if z_front.size > 0:
                            depth_val = float(np.median(z_front))
                        else:
                            # fallback: median range if all non-front
                            depth_val = float(np.median(np.linalg.norm(verts[m], axis=1)))
                        if (best_depth is None) or (depth_val < best_depth):
                            best_depth = depth_val
                            best_label = lab

                    if best_label is None:
                        # fallback to largest cluster
                        counts = np.bincount(labels[labels >= 0])
                        best_label = int(np.argmax(counts))

                    keep_mask = (labels == best_label)
                    verts_f = verts[keep_mask]
                    tex_f = tex[keep_mask]

            except Exception as e:
                if getattr(self, "debug", False):
                    print(f"[nearest_cluster] DBSCAN failed: {e}")
                # fall back to keeping all (keep_mask already all-True)

        return verts_f, tex_f


    def _auto_eps(self, pts: np.ndarray, sample: int = 200) -> float:
        """Heuristic DBSCAN eps from the median nearest-neighbor distance of a random subset."""
        n = len(pts)
        if n < 3:
            return 0.01
        idx = np.random.choice(n, size=min(sample, n), replace=False)
        ps = pts[idx]
        D = np.linalg.norm(ps[:, None, :] - ps[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        nn = D.min(axis=1)
        eps = 1.5 * np.median(nn)
        return float(max(eps, 0.005))


    def _voxel_density_mask(self, pts: np.ndarray, voxel: float = 0.01, min_points_per_voxel: int = 4) -> np.ndarray:
        """Boolean mask keeping points in voxels with at least N points."""
        if len(pts) == 0:
            return np.zeros(0, dtype=bool)
        keys = np.floor(pts / voxel).astype(np.int32)
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        return counts[inv] >= int(min_points_per_voxel)
