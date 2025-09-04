import pyrealsense2 as rs
import numpy as np
import cv2
import time
import open3d as o3d

class RealSenseCamera:
    """
    Wrapper for Intel RealSense D435i RGB-D camera. Provides synchronized color/depth frames,
    intrinsics, pointcloud generation, and streaming utilities.
    """
    def __init__(self, width=1280, height=720, fps=30, align_to='color', enable_filters=True, near_mode=True):
        """
        Initialize the RealSense pipeline and configure streams.

        Args:
            width (int): Frame width in pixels.
            height (int): Frame height in pixels.
            fps (int): Frame rate.
            align_to (str or None): 'color', 'depth', or None, to align depth to color or vice versa.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_filters = enable_filters
        self.near_mode = near_mode

        # Configure and start pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(config)

        # Alignment setup
        if align_to == 'color':
            # Warp the depth image so that its pixels line up with the color image
            self.align = rs.align(rs.stream.color)
        elif align_to == 'depth':
            # Warp the color image so that it lines up with the depth image
            self.align = rs.align(rs.stream.depth)
        else:
            self.align = None

        # Depth sensor settings
        dev = self.profile.get_device()
        depth_sensor = dev.first_depth_sensor()

        # change the mode of D435i camera, modes include:
        # default/high_density/medium_density/high_accuracy/hand
        time.sleep(0.5)
        if near_mode and depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(
                rs.option.visual_preset,
                rs.rs400_visual_preset.default)
        self.depth_scale = depth_sensor.get_depth_scale()

        # Get intrinsics
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.color_intrinsics = color_profile.get_intrinsics()
        self.depth_intrinsics = depth_profile.get_intrinsics()

        # Setup filters
        if self.enable_filters:
            self.hole_filling = rs.hole_filling_filter(mode = 0) # 0: fill_from_left; 1: farest_from_around; 2: nearest_from_around
            
        # Pointcloud object
        self.pc = rs.pointcloud()

    def get_frames(self, raw =False):
        """
        Capture a pair of aligned color and depth frames.

        Returns:
            color_image (ndarray) or None
            depth_image (ndarray) or None
        """
        try:
            frames = self.pipeline.wait_for_frames()
        except RuntimeError as e:
            print(f"Warning: could not get frames: {e}")
            return None, None
        if self.align:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if raw:
            return color_frame, depth_frame
        if not depth_frame or not color_frame:
            return None, None

        # filter out None values
        if self.enable_filters:
            depth_frame = self.hole_filling.process(depth_frame)
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

    def get_intrinsics(self):
        """
        Retrieve camera intrinsics and depth scale.
        """
        return {
            'color': self.color_intrinsics,
            'depth': self.depth_intrinsics,
            'depth_scale': self.depth_scale
        }

    def get_pointcloud(self,
                   bbox=None,
                   max_points=None,           # cap number of returned points
                   voxel=0.01,                 # if given -> use voxel downsampling
                   seed=998,
                   fps_presample=5):           # pre-sample ratio for FPS speedup
        """
        Generate a point cloud from latest frames.
        If bbox is provided, only back-project pixels within the bounding box (fast ROI mode).
        Downsampling:
        - If voxel is not None: 3D voxel downsample (keep one point/voxel)
        - Else: Furthest Point Sampling (FPS), optionally with pre-sampling.

        Args:
            bbox: (x, y, w, h) in pixel coords (color image).
            max_points: int or None. If None or >= N, returns all points.
            voxel: voxel size (meters) for voxel downsampling. If None -> FPS.
            seed: RNG seed (for voxel & fallback tops-ups).
            fps_presample: if N > fps_presample * max_points, randomly pre-sample that many
                        to make FPS O((fps_presample*max_points)*max_points), much faster.

        Returns:
            verts: (K,3) float32 camera-frame points (meters)
            tex:   (K,2) float32 normalized texture coords (u,v in [0,1])
        """

        def _fps_indices(pts, target, seed=None, presample=5):
            """Furthest Point Sampling (Euclidean). Returns indices into pts."""
            rng = np.random.default_rng(seed)
            N = len(pts)
            if target >= N:
                return np.arange(N, dtype=np.int64)

            # optional pre-sample to reduce complexity
            if presample is not None and presample > 1 and N > presample * target:
                pool = rng.choice(N, size=presample * target, replace=False)
                P = pts[pool]
                pool_idx = pool
            else:
                P = pts
                pool_idx = np.arange(N, dtype=np.int64)

            K = min(target, len(P))
            sel_local = np.empty(K, dtype=np.int64)

            # init: farthest from mean (more stable than random)
            mean = P.mean(axis=0)
            d2 = np.sum((P - mean) ** 2, axis=1)
            sel_local[0] = int(np.argmax(d2))

            # maintain min distance to current set (squared distances for speed)
            dist2 = np.sum((P - P[sel_local[0]]) ** 2, axis=1)
            for i in range(1, K):
                sel_local[i] = int(np.argmax(dist2))
                newd2 = np.sum((P - P[sel_local[i]]) ** 2, axis=1)
                # update the running nearest distance to selected set
                dist2 = np.minimum(dist2, newd2)

            return pool_idx[sel_local]

        def _voxel_indices(pts, target, voxel, seed=None):
            """One point per 3D voxel (first occurrence), then trim/pad to target."""
            rng = np.random.default_rng(seed)
            N = len(pts)
            if target >= N:
                return np.arange(N, dtype=np.int64)
            if voxel is None:
                voxel = 0.01  # 1 cm default
            keys = np.floor(pts / float(voxel)).astype(np.int64)
            try:
                _, first_idx = np.unique(keys, axis=0, return_index=True)
                sel = first_idx
            except TypeError:
                seen = {}
                idxs = []
                for i, k in enumerate(map(tuple, keys)):
                    if k not in seen:
                        seen[k] = True
                        idxs.append(i)
                sel = np.asarray(idxs, dtype=np.int64)

            if len(sel) > target:
                sel = rng.choice(sel, size=target, replace=False)
            elif len(sel) < target:
                remaining = np.setdiff1d(np.arange(N, dtype=np.int64), sel, assume_unique=False)
                need = target - len(sel)
                if len(remaining) > 0 and need > 0:
                    extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
                    sel = np.concatenate([sel, extra], axis=0)
            return sel

        def _downsample(pts, tex, target, voxel, seed, presample):
            """Choose indices according to voxel vs FPS, return downsampled pts/tex."""
            if target is None or target >= len(pts):
                return pts, tex
            if voxel is not None:
                sel = _voxel_indices(pts, int(target), voxel, seed)
            else:
                sel = _fps_indices(pts, int(target), seed=seed, presample=presample)
            sel.sort()  # stable order
            return pts[sel], tex[sel]

        # ----- ROI back-projection path -----
        if bbox is not None:
            color_image, depth_image = self.get_frames(raw=False)
            if color_image is None or depth_image is None:
                return None, None

            x, y, w, h = map(int, bbox)
            H, W = depth_image.shape
            x0 = max(0, x); y0 = max(0, y)
            x1 = min(W, x + w); y1 = min(H, y + h)
            if x1 <= x0 or y1 <= y0:
                return None, None

            depth_roi = depth_image[y0:y1, x0:x1]
            us, vs = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            us = us.ravel(); vs = vs.ravel()

            ds = depth_roi.ravel().astype(np.float32) * self.depth_scale
            valid = ds > 0
            if not np.any(valid):
                return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32)

            us, vs, ds = us[valid], vs[valid], ds[valid]

            fx, fy = self.color_intrinsics.fx, self.color_intrinsics.fy
            cx, cy = self.color_intrinsics.ppx, self.color_intrinsics.ppy

            xs = (us - cx) * ds / fx
            ys = (vs - cy) * ds / fy
            pts = np.vstack((xs, ys, ds)).T.astype(np.float32)

            tex = np.vstack((us.astype(np.float32) / W,
                            vs.astype(np.float32) / H)).T.astype(np.float32)

            pts, tex = _downsample(pts, tex, max_points, voxel, seed, fps_presample)
            return pts, tex

        # ----- full-pointcloud path (SDK) -----
        color_frame, depth_frame = self.get_frames(raw=True)
        if color_frame is None or depth_frame is None:
            return None, None

        self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        tex   = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

        verts, tex = _downsample(verts, tex, max_points, voxel, seed, fps_presample)
        return verts, tex


    def stream_color(self):
        """
        Generator that yields the latest color frames.
        """
        while True:
            color, _ = self.get_frames()
            if color is None:
                break
            yield color

    def stream_depth(self):
        """
        Generator that yields the latest depth frames.
        """
        while True:
            _, depth = self.get_frames()
            if depth is None:
                break
            yield depth

    def stream_frames(self):
        """
        Generator that yields synchronized (color, depth) frame pairs.
        """
        while True:
            color, depth = self.get_frames()
            if color is None or depth is None:
                break
            yield color, depth

    def stop(self):
        """
        Stop the camera pipeline.
        """
        self.pipeline.stop()

    def __del__(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class PointCloudVisualizer:
    """
    Visualizer for live updating of point clouds using Open3D.
    """
    def __init__(self, camera: RealSenseCamera, window_name='Live PointCloud'):
        self.camera = camera
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name)
        self.pcd = o3d.geometry.PointCloud()
        self.added = False

        # Style knobs
        ro = self.vis.get_render_option()
        ro.point_size = 3.0                 # larger cloud points
        # ro.line_width = 2.0               # only if your Open3D build supports it
        self.cloud_gray = 0.35              # set to None to keep texture colors
        self.voxel_size = 0.0               # e.g., 0.01 (meters) to downsample

        # Curve layers
        self.curve_ls = None                # fallback polyline
        self.curve_mesh = None              # thick tube version
        self.curve_added = False
        self.ctrl_pcd = None                # small points (fallback)
        self.ctrl_mesh = None               # spheres for control pts
        self.ctrl_added = False

        # Mesh styling
        self.curve_radius = 0.004           # meters (tube radius)
        self.ctrl_radius  = 0.006           # meters (sphere radius)
        self.curve_color  = (0.0, 1.0, 1.0) # yellow

        # Sampling pose styling 
        self.sample_radius = 0.008                # sphere radius (m)
        self.sample_color  = (0.0, 0.0, 1.0)      # magenta
        self.arrow_color   = (0.0, 0.0, 1.0)      # magenta
        self.arrow_len     = 0.08                 # default arrow length (m)
        self.arrow_cyl_r   = 0.0025               # cylinder radius (m)
        self.arrow_cone_r  = 0.004                # cone radius (m)
        self.arrow_cone_hf = 0.3                  # cone is 30% of total length

        self.sample_mesh = None                   # sphere at sample point
        self.arrow_mesh  = None                   # arrow for approach dir

    def update(self, verts, tex):
        """
        Update the visualizer with a new point cloud frame.
        """
        color_image, _ = self.camera.get_frames()
        if color_image is None:
            return
        if verts is None or len(verts) == 0:
            # still poll so the window stays responsive
            self.vis.poll_events(); self.vis.update_renderer()
            return

        # Update point positions and colors
        self.pcd.points = o3d.utility.Vector3dVector(verts)
        H, W, _ = color_image.shape
        u = np.clip((tex[:,0]*W).astype(int), 0, W-1)
        v = np.clip((tex[:,1]*H).astype(int), 0, H-1)
        colors = color_image[v, u] / 255.0
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        # Add or refresh geometry
        if not self.added:
            self.vis.add_geometry(self.pcd)
            self.added = True
        else:
            self.vis.update_geometry(self.pcd)
        if self.sample_mesh is not None:
            self.vis.update_geometry(self.sample_mesh)
        if self.arrow_mesh is not None:
            self.vis.update_geometry(self.arrow_mesh)
            

        self.vis.poll_events()
        self.vis.update_renderer()

    def set_curve(self, curve_pts, use_tube=True, color=None, radius=None):
        """
        Create/update a smooth curve in the scene.
        - use_tube=True: draw as a thick tube (TriangleMesh) for maximum visibility
        - use_tube=False: fallback to a simple LineSet polyline
        """
        if curve_pts is None or len(curve_pts) < 2:
            return
        curve_pts = np.asarray(curve_pts, dtype=np.float32)
        color = self.curve_color if color is None else color
        radius = self.curve_radius if radius is None else radius

        if use_tube:
            mesh = self._polyline_to_tube(curve_pts, radius, color)
            # first time: add; else: replace geometry efficiently
            if self.curve_mesh is None:
                self.curve_mesh = mesh
                self.vis.add_geometry(self.curve_mesh)
            else:
                # replace vertices/triangles in place
                self.curve_mesh.vertices  = mesh.vertices
                self.curve_mesh.triangles = mesh.triangles
                self.curve_mesh.vertex_colors = mesh.vertex_colors
                self.vis.update_geometry(self.curve_mesh)
        else:
            if self.curve_ls is None:
                self.curve_ls = o3d.geometry.LineSet()
                self.vis.add_geometry(self.curve_ls)
            self.curve_ls.points = o3d.utility.Vector3dVector(curve_pts)
            lines = np.column_stack([np.arange(len(curve_pts) - 1),
                                     np.arange(1, len(curve_pts))]).astype(np.int32)
            self.curve_ls.lines = o3d.utility.Vector2iVector(lines)
            self.curve_ls.colors = o3d.utility.Vector3dVector(
                np.tile(np.asarray(color, dtype=np.float32), (len(lines), 1))
            )
            self.vis.update_geometry(self.curve_ls)

    def set_control(self, ctrl_pts, use_spheres=True, color=None, radius=None):
        """
        Create/update a control/median point layer.
        - use_spheres=True: draw as spheres (TriangleMesh)
        - use_spheres=False: fallback to small points
        """
        if ctrl_pts is None or len(ctrl_pts) == 0:
            return
        ctrl_pts = np.asarray(ctrl_pts, dtype=np.float32)
        color = self.ctrl_color if color is None else color
        radius = self.ctrl_radius if radius is None else radius

        if use_spheres:
            mesh = self._points_to_spheres(ctrl_pts, radius, color)
            if self.ctrl_mesh is None:
                self.ctrl_mesh = mesh
                self.vis.add_geometry(self.ctrl_mesh)
            else:
                self.ctrl_mesh.vertices  = mesh.vertices
                self.ctrl_mesh.triangles = mesh.triangles
                self.ctrl_mesh.vertex_colors = mesh.vertex_colors
                self.vis.update_geometry(self.ctrl_mesh)
        else:
            if self.ctrl_pcd is None:
                self.ctrl_pcd = o3d.geometry.PointCloud()
                self.vis.add_geometry(self.ctrl_pcd)
            self.ctrl_pcd.points = o3d.utility.Vector3dVector(ctrl_pts)
            self.ctrl_pcd.colors = o3d.utility.Vector3dVector(
                np.tile(np.asarray(color, dtype=np.float32), (len(ctrl_pts), 1))
            )
            self.vis.update_geometry(self.ctrl_pcd)

    def clear_curve(self):
        """Remove curve/control meshes from the scene."""
        if self.curve_mesh is not None:
            self.vis.remove_geometry(self.curve_mesh, reset_bounding_box=False)
        if self.curve_ls is not None:
            self.vis.remove_geometry(self.curve_ls, reset_bounding_box=False)
        if self.ctrl_mesh is not None:
            self.vis.remove_geometry(self.ctrl_mesh, reset_bounding_box=False)
        if self.ctrl_pcd is not None:
            self.vis.remove_geometry(self.ctrl_pcd, reset_bounding_box=False)
        self.curve_mesh = self.curve_ls = self.ctrl_mesh = self.ctrl_pcd = None

    def set_sample(self, point, approach_dir=None,
                   point_radius=None, point_color=None,
                   arrow_len=None, arrow_color=None,
                   head_at_point=False):
        """
        Draw / update the sampling point (sphere) and optional approach arrow.

        Args:
            point         : (3,) in camera frame (meters)
            approach_dir  : (3,) unit-ish vector; if None, only the sphere is drawn
            point_radius  : overrides self.sample_radius
            point_color   : (r,g,b) in [0,1]
            arrow_len     : arrow length in meters (default self.arrow_len)
            arrow_color   : (r,g,b) in [0,1]
            head_at_point : if True, the arrow head sits on the sample; otherwise head points away
        """
        if point is None or len(point) != 3:
            return

        p = np.asarray(point, dtype=np.float32)
        pr = self.sample_radius if point_radius is None else float(point_radius)
        pc = self.sample_color  if point_color  is None else tuple(point_color)

        # --- sample sphere ---
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=pr, resolution=16)
        sph.translate(p)
        sph.compute_vertex_normals()
        sph.paint_uniform_color(np.asarray(pc, dtype=np.float32))

        if self.sample_mesh is None:
            self.sample_mesh = sph
            self.vis.add_geometry(self.sample_mesh)
        else:
            self.sample_mesh.vertices = sph.vertices
            self.sample_mesh.triangles = sph.triangles
            self.sample_mesh.vertex_colors = sph.vertex_colors
            self.sample_mesh.compute_vertex_normals()
            self.vis.update_geometry(self.sample_mesh)

        # --- approach arrow (optional) ---
        if approach_dir is None:
            return

        a = np.asarray(approach_dir, dtype=np.float32)
        na = float(np.linalg.norm(a))
        if na < 1e-6:
            return
        a = a / na

        L  = self.arrow_len if arrow_len is None else float(arrow_len)
        L  = max(L, 1e-3)
        Lc = max(L * (1.0 - self.arrow_cone_hf), 1e-3)   # cylinder height
        Ln = max(L - Lc, 1e-3)                           # cone height
        rc = float(self.arrow_cyl_r)
        rn = float(self.arrow_cone_r)
        col = np.asarray(self.arrow_color if arrow_color is None else arrow_color, dtype=np.float32)

        arr = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=rc, cone_radius=rn,
                                                     cylinder_height=Lc, cone_height=Ln,
                                                     resolution=20, cylinder_split=1, cone_split=1)
        # Arrow is along +Z; rotate to 'a'
        R = self._rotation_matrix_from_vectors(np.array([0.0,0.0,1.0], dtype=np.float32), a)
        arr.rotate(R, center=np.zeros(3))

        # place arrow
        # - If head_at_point=False: base at sample, head away from it
        # - If head_at_point=True : tip at sample (shift back along -a)
        if head_at_point:
            arr.translate(p - a * L)
        else:
            arr.translate(p)

        arr.compute_vertex_normals()
        arr.paint_uniform_color(col)

        if self.arrow_mesh is None:
            self.arrow_mesh = arr
            self.vis.add_geometry(self.arrow_mesh)
        else:
            self.arrow_mesh.vertices = arr.vertices
            self.arrow_mesh.triangles = arr.triangles
            self.arrow_mesh.vertex_colors = arr.vertex_colors
            self.arrow_mesh.compute_vertex_normals()
            self.vis.update_geometry(self.arrow_mesh)

    def clear_sample(self):
        """Remove sampling point & arrow from the scene."""
        if self.sample_mesh is not None:
            self.vis.remove_geometry(self.sample_mesh, reset_bounding_box=False)
            self.sample_mesh = None
        if self.arrow_mesh is not None:
            self.vis.remove_geometry(self.arrow_mesh, reset_bounding_box=False)
            self.arrow_mesh = None

    def _polyline_to_tube(self, pts, radius, color, seg_res=16):
        """
        Sweep small cylinders along the polyline; merge into one TriangleMesh.
        """
        eps = 1e-8
        color = np.asarray(color, dtype=np.float32)
        tube = o3d.geometry.TriangleMesh()
        tube.vertices = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        tube.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))
        base_vert_count = 0

        zaxis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            d = p2 - p1
            L = np.linalg.norm(d)
            if L < eps:
                continue
            d_hat = d / L

            # cylinder is along +Z; rotate to d_hat
            cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L, resolution=seg_res)
            R = self._rotation_matrix_from_vectors(zaxis, d_hat)
            cyl.rotate(R, center=np.zeros(3))
            # move to segment midpoint
            cyl.translate((p1 + p2) * 0.5)
            cyl.compute_vertex_normals()
            cyl.paint_uniform_color(color)

            # merge into tube
            v = np.asarray(cyl.vertices)
            f = np.asarray(cyl.triangles, dtype=np.int32)
            if len(tube.vertices) == 0:
                tube.vertices = o3d.utility.Vector3dVector(v)
                tube.triangles = o3d.utility.Vector3iVector(f)
            else:
                tv = np.asarray(tube.vertices)
                tf = np.asarray(tube.triangles, dtype=np.int32)
                tube.vertices  = o3d.utility.Vector3dVector(np.vstack([tv, v]))
                tube.triangles = o3d.utility.Vector3iVector(np.vstack([tf, f + len(tv)]))

        tube.paint_uniform_color(color)
        return tube

    def _points_to_spheres(self, pts, radius, color, res=12):
        """
        Create a single TriangleMesh of spheres placed at pts.
        """
        color = np.asarray(color, dtype=np.float32)
        all_mesh = o3d.geometry.TriangleMesh()
        all_mesh.vertices = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        all_mesh.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))

        for p in pts:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=res)
            s.translate(p)
            s.compute_vertex_normals()
            s.paint_uniform_color(color)

            v = np.asarray(s.vertices)
            f = np.asarray(s.triangles, dtype=np.int32)
            if len(all_mesh.vertices) == 0:
                all_mesh.vertices = o3d.utility.Vector3dVector(v)
                all_mesh.triangles = o3d.utility.Vector3iVector(f)
            else:
                tv = np.asarray(all_mesh.vertices)
                tf = np.asarray(all_mesh.triangles, dtype=np.int32)
                all_mesh.vertices  = o3d.utility.Vector3dVector(np.vstack([tv, v]))
                all_mesh.triangles = o3d.utility.Vector3iVector(np.vstack([tf, f + len(tv)]))

        all_mesh.paint_uniform_color(color)
        return all_mesh

    @staticmethod
    def _rotation_matrix_from_vectors(a, b, eps=1e-8):
        """
        Rotation matrix that rotates unit vector a to unit vector b.
        """
        a = a / (np.linalg.norm(a) + eps)
        b = b / (np.linalg.norm(b) + eps)
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        if np.linalg.norm(v) < eps:
            # parallel or anti-parallel
            if c > 0.0:
                return np.eye(3)
            # 180-degree turn around any orthogonal axis
            axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            v = np.cross(a, axis)
            v /= (np.linalg.norm(v) + eps)
            K = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
            return np.eye(3) + 2 * K @ K  # Rodrigues with theta=pi
        K = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
        return np.eye(3) + K + K @ K * (1.0 / (1.0 + c + eps))
    
    def close(self):
        """Close the visualizer window."""
        self.vis.destroy_window()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class BranchCurveFitter:
    def __init__(self,
                 nbins: int = 24,
                 alpha: float = 0.5,
                 min_per_bin: int = 8,
                 ema_ctrl: float = 0.25,   # smoothing for control points (0.1–0.35)
                 ema_axis: float = 0.20,   # smoothing for PCA axis & center
                 ema_span: float = 0.25,   # smoothing for tmin/tmax span
                 ):
        self.nbins = nbins
        self.alpha = alpha
        self.min_per_bin = min_per_bin
        self.ema_ctrl = ema_ctrl
        self.ema_axis = ema_axis
        self.ema_span = ema_span

        # ---- temporal state ----
        self._v1 = None        # EMA principal axis (unit)
        self._c  = None        # EMA center
        self._tmin = None      # EMA tmin along axis
        self._tmax = None      # EMA tmax along axis
        self._ctrl_ema = None  # (nbins,3) EMA control points

    def fit(self, pts: np.ndarray, nbins: int = None, samples: int = 100,
            min_per_bin: int = None, alpha: float = None):
        """
        Temporal/stable fit:
          - EMA PCA axis/center with sign-consistency
          - EMA tmin/tmax -> fixed, stable bins
          - Per-bin median -> EMA control points (fallback to previous if empty)
        """
        if pts is None or len(pts) < 3 * (min_per_bin or self.min_per_bin):
            return None
        P = np.asarray(pts, dtype=np.float32)

        nb = self.nbins if nbins is None else int(nbins)
        mpb = self.min_per_bin if min_per_bin is None else int(min_per_bin)
        a  = self.alpha if alpha is None else float(alpha)

        # ---- 1) instantaneous PCA (for this frame) ----
        c_cur, v1_cur = self._pca_axis(P)

        # sign consistency w.r.t. previous axis
        if self._v1 is not None and np.dot(v1_cur, self._v1) < 0:
            v1_cur = -v1_cur

        # ---- 2) EMA axis & center ----
        if self._v1 is None:
            self._v1 = v1_cur.copy()
            self._c  = c_cur.copy()
        else:
            self._v1 = self._unit((1.0 - self.ema_axis) * self._v1 + self.ema_axis * v1_cur)
            self._c  = (1.0 - self.ema_axis) * self._c + self.ema_axis * c_cur

        # project onto *smoothed* axis
        t = (P - self._c) @ self._v1
        tmin_cur, tmax_cur = float(np.min(t)), float(np.max(t))

        # ---- 3) EMA span -> stable bins ----
        if self._tmin is None:
            self._tmin, self._tmax = tmin_cur, tmax_cur
        else:
            self._tmin = (1.0 - self.ema_span) * self._tmin + self.ema_span * tmin_cur
            self._tmax = (1.0 - self.ema_span) * self._tmax + self.ema_span * tmax_cur
            # guard against collapse
            if self._tmax - self._tmin < 1e-4:
                pad = 5e-4
                self._tmin -= pad
                self._tmax += pad

        edges = np.linspace(self._tmin, self._tmax, nb + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # ---- 4) per-bin median (this frame) ----
        med = np.full((nb, 3), np.nan, dtype=np.float32)
        for i in range(nb):
            m = (t >= edges[i]) & (t < edges[i+1]) if i < nb-1 else (t >= edges[i]) & (t <= edges[i+1])
            if np.count_nonzero(m) >= mpb:
                med[i] = np.median(P[m], axis=0)

        # If first frame: initialize ctrl by interpolating through available medians
        if self._ctrl_ema is None:
            valid = ~np.isnan(med[:, 0])
            if np.count_nonzero(valid) < 4:
                return None
            # linear interp across bins for each coord
            idx = np.where(valid)[0]
            ctrl0 = np.empty_like(med)
            for d in range(3):
                ctrl0[:, d] = np.interp(np.arange(nb), idx, med[idx, d])
            self._ctrl_ema = ctrl0
        else:
            # update EMA only where we have new medians; keep previous elsewhere
            upd = ~np.isnan(med[:, 0])
            self._ctrl_ema[upd] = (1.0 - self.ema_ctrl) * self._ctrl_ema[upd] + self.ema_ctrl * med[upd]
            # (bins without new data just carry over previous EMA)

        # ---- 5) deduplicate/clean, then spline ----
        ctrl = self._ctrl_ema.copy()
        # remove consecutive near-duplicates
        keep = [0]
        for i in range(1, len(ctrl)):
            if np.linalg.norm(ctrl[i] - ctrl[keep[-1]]) > 1e-6:
                keep.append(i)
        ctrl = ctrl[keep]
        if len(ctrl) < 4:
            return None

        curve = self._catmull_rom(ctrl, samples=samples, alpha=a)
        tangent = self._finite_diff_unit(curve)
        curvature = self._curvature(curve)
        length = float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))

        return {
            'ctrl': ctrl.astype(np.float32),
            'curve': curve.astype(np.float32),
            'tangent': tangent.astype(np.float32),
            'curvature': curvature.astype(np.float32),
            'length': length,
        }

    def fit_stiff(self, pts: np.ndarray, samples: int = 100,
              deg: int = 3, lam: float = 1e-2, ema_coef: float = 0.2):
        """
        Stiff curve fit: cubic (or quadratic) polynomial with ridge regularization.
        Keeps EMA of PCA axis/center and polynomial coefficients across frames.

        Returns same dict keys as fit(), but 'ctrl' are just sample anchors.
        """
        if pts is None or len(pts) < 40:
            return None
        P = np.asarray(pts, dtype=np.float32)

        # 1) smooth PCA axis/center (reuse your existing EMA + sign-consistency + axis cap from patch A)
        c_cur, v1_cur = self._pca_axis(P)
        if self._v1 is not None and np.dot(v1_cur, self._v1) < 0:
            v1_cur = -v1_cur
        # optional: apply same max-axis-step cap here as in patch A
        if self._v1 is None:
            self._v1 = v1_cur.copy(); self._c = c_cur.copy()
        else:
            self._v1 = self._unit((1.0 - self.ema_axis) * self._v1 + self.ema_axis * v1_cur)
            self._c  = (1.0 - self.ema_axis) * self._c  + self.ema_axis * c_cur

        # 2) project to t along axis; normalize t to [0,1] via EMA min/max
        t = (P - self._c) @ self._v1
        tmin_cur, tmax_cur = float(np.min(t)), float(np.max(t))
        if self._tmin is None:
            self._tmin, self._tmax = tmin_cur, tmax_cur
        else:
            self._tmin = (1.0 - self.ema_span) * self._tmin + self.ema_span * tmin_cur
            self._tmax = (1.0 - self.ema_span) * self._tmax + self.ema_span * tmax_cur
            if self._tmax - self._tmin < 1e-4:
                pad = 5e-4; self._tmin -= pad; self._tmax += pad

        tt = (t - self._tmin) / (self._tmax - self._tmin + 1e-9)   # ~ [0,1]
        # design matrix
        deg = int(deg)
        Phi = np.vstack([tt**k for k in range(deg+1)]).T  # (N, deg+1)
        # ridge solve per coordinate: (Phi^T Phi + lam I) beta = Phi^T y
        I = np.eye(deg+1, dtype=np.float32) * lam
        A = (Phi.T @ Phi + I)

        def solve_coef(y, beta_prev=None):
            rhs = Phi.T @ y
            beta = np.linalg.solve(A, rhs)
            if beta_prev is None:
                return beta
            return (1.0 - ema_coef) * beta_prev + ema_coef * beta

        # keep EMA of coefficients across frames
        if not hasattr(self, "_coef"):
            self._coef = {}

        bx_prev = self._coef.get('x'); by_prev = self._coef.get('y'); bz_prev = self._coef.get('z')
        bx = solve_coef(P[:,0], bx_prev)
        by = solve_coef(P[:,1], by_prev)
        bz = solve_coef(P[:,2], bz_prev)
        self._coef['x'], self._coef['y'], self._coef['z'] = bx, by, bz

        # 3) sample curve uniformly in s∈[0,1]
        s = np.linspace(0.0, 1.0, samples, endpoint=True).astype(np.float32)
        S = np.vstack([s**k for k in range(deg+1)]).T
        X = S @ bx; Y = S @ by; Z = S @ bz
        curve = np.stack([X, Y, Z], axis=1)

        # 4) tangents/curvature/length as before
        tangent = self._finite_diff_unit(curve)
        curvature = self._curvature(curve)
        length = float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))

        # provide ctrl anchors (optional): just every few samples
        ctrl_idx = np.linspace(0, samples-1, num=min(8, samples), dtype=int)
        ctrl = curve[ctrl_idx]

        return {
            'ctrl': ctrl.astype(np.float32),
            'curve': curve.astype(np.float32),
            'tangent': tangent.astype(np.float32),
            'curvature': curvature.astype(np.float32),
            'length': length,
        }
    
    @staticmethod
    def _unit(v, eps=1e-9):
        n = np.linalg.norm(v)
        return v / (n + eps)

    # ---------- helpers ----------

    def _pca_axis(self, pts):
        c = np.mean(pts, axis=0)
        X = pts - c
        C = (X.T @ X) / max(len(pts) - 1, 1)
        evals, evecs = np.linalg.eigh(C)  # ascending
        v1 = evecs[:, -1]
        v1 /= (np.linalg.norm(v1) + 1e-9)
        return c, v1

    def _catmull_rom(self, P, samples=100, alpha=0.5, eps=1e-6):
        """
        Centripetal Catmull–Rom through control points P (K,3).
        Returns S sampled points.
        """
        P = np.asarray(P, dtype=np.float32)
        # 1) Deduplicate consecutive points that are too close
        keep = [0]
        for i in range(1, len(P)):
            if np.linalg.norm(P[i] - P[keep[-1]]) > eps:
                keep.append(i)
        P = P[keep]
        K = len(P)
        if K < 2:
            return P.copy()

        # segments are [0..K-2]
        nseg = K - 1
        samples_per = max(2, samples // nseg)

        curve = []
        for i in range(0, nseg):
            P0 = P[max(i - 1, 0)]
            P1 = P[i]
            P2 = P[i + 1]
            P3 = P[min(i + 2, K - 1)]

            # Centripetal parameterization
            def tj(ti, Pi, Pj):
                d = np.linalg.norm(Pj - Pi)
                if d < eps:
                    d = eps
                return ti + d**alpha

            t0 = 0.0
            t1 = tj(t0, P0, P1)
            t2 = tj(t1, P1, P2)
            t3 = tj(t2, P2, P3)

            # If the segment is degenerate, fall back to uniform Catmull–Rom
            use_uniform = (t1 - t0 <= eps) or (t2 - t1 <= eps) or (t3 - t2 <= eps)

            if use_uniform:
                uvals = np.linspace(0.0, 1.0, samples_per, endpoint=(i == nseg - 1))
                for u in uvals:
                    # Uniform CR spline (cardinal form with tension = 0.5)
                    m1 = 0.5 * (P2 - P0)
                    m2 = 0.5 * (P3 - P1)
                    u2 = u * u
                    u3 = u2 * u
                    C = (2*u3 - 3*u2 + 1) * P1 + (u3 - 2*u2 + u) * m1 \
                        + (-2*u3 + 3*u2) * P2 + (u3 - u2) * m2
                    curve.append(C)
            else:
                ts = np.linspace(t1, t2, samples_per, endpoint=(i == nseg - 1))
                for t in ts:
                    A1 = ((t1 - t) / (t1 - t0 + eps)) * P0 + ((t - t0) / (t1 - t0 + eps)) * P1
                    A2 = ((t2 - t) / (t2 - t1 + eps)) * P1 + ((t - t1) / (t2 - t1 + eps)) * P2
                    A3 = ((t3 - t) / (t3 - t2 + eps)) * P2 + ((t - t2) / (t3 - t2 + eps)) * P3
                    B1 = ((t2 - t) / (t2 - t0 + eps)) * A1 + ((t - t0) / (t2 - t0 + eps)) * A2
                    B2 = ((t3 - t) / (t3 - t1 + eps)) * A2 + ((t - t1) / (t3 - t1 + eps)) * A3
                    C  = ((t2 - t) / (t2 - t1 + eps)) * B1 + ((t - t1) / (t2 - t1 + eps)) * B2
                    curve.append(C)

        return np.vstack(curve)

    def _finite_diff_unit(self, curve):
        d = np.gradient(curve, axis=0)
        n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        return d / n

    def _curvature(self, curve):
        # κ = ||dT/ds|| with T = unit tangent
        T = self._finite_diff_unit(curve)
        dT = np.gradient(T, axis=0)
        ds = np.linalg.norm(np.gradient(curve, axis=0), axis=1) + 1e-9
        kappa = np.linalg.norm(dT, axis=1) / ds
        return kappa
    
class PoseSmoother:
    """
    Temporal smoother for (point, direction) only.
    - Point: exponential moving average (EMA)
    - Direction: SLERP with per-frame rotation cap and sign consistency
    """
    def __init__(self, pos_alpha=0.20, dir_alpha=0.25, max_rot_deg=20.0, keep_sign=True):
        self.pos_alpha = float(pos_alpha)     # 0.1–0.3 -> heavier smoothing
        self.dir_alpha = float(dir_alpha)     # SLERP step per frame (0..1)
        self.max_rot = np.deg2rad(max_rot_deg)
        self.keep_sign = bool(keep_sign)

        self._P = None        # EMA point
        self._D = None        # smoothed direction (unit)
        self._t_prev = None   # last timestamp

    @staticmethod
    def _unit(v, eps=1e-9):
        if v is None:
            return None
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return (v / n) if n >= eps else None

    @staticmethod
    def _slerp(u, v, alpha, eps=1e-8):
        u = PoseSmoother._unit(u)
        v = PoseSmoother._unit(v)
        if u is None and v is None:
            return None
        if u is None:
            return v
        if v is None:
            return u
        dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
        if 1.0 - abs(dot) < 1e-6:
            # nearly parallel/antiparallel -> lerp toward aligned target, then renorm
            v_aligned = v if dot > 0 else -v
            w = (1.0 - alpha) * u + alpha * v_aligned
            res = PoseSmoother._unit(w)
            return res if res is not None else u
        theta = np.arccos(dot)
        s = np.sin(theta)
        return (np.sin((1.0 - alpha) * theta) / s) * u + (np.sin(alpha * theta) / s) * v
    
    def update(self, P_raw, D_raw):
        now = time.time()
        P_raw = np.asarray(P_raw, dtype=np.float32)
        D_raw_u = self._unit(D_raw)

        # init
        if self._P is None or self._D is None:
            self._P = P_raw.copy()
            self._D = D_raw_u if D_raw_u is not None else np.array([0, 0, 1], dtype=np.float32)
            self._t_prev = now
            return self._P.copy(), self._D.copy()

        # point EMA
        a = float(self.pos_alpha)
        self._P = (1.0 - a) * self._P + a * P_raw

        # direction: keep previous if new invalid
        if D_raw_u is None:
            self._t_prev = now
            return self._P.copy(), self._D.copy()

        # sign consistency
        if self.keep_sign and np.dot(D_raw_u, self._D) < 0:
            D_raw_u = -D_raw_u

        # rotation cap + SLERP
        cosang = float(np.clip(np.dot(self._D, D_raw_u), -1.0, 1.0))
        ang = np.arccos(cosang)
        if ang < 1e-6:
            D_new = self._D
        else:
            step = min(self.dir_alpha, self.max_rot / ang)
            D_new = self._slerp(self._D, D_raw_u, step)
            if D_new is None:
                D_new = self._D

        d_norm = self._unit(D_new)
        self._D = d_norm if d_norm is not None else self._D
        self._t_prev = now
        return self._P.copy(), self._D.copy()
    
# Example usage:
'''cam = RealSenseCamera(width=640, height=480, fps=30, align_to='color')
color, depth = cam.get_frames()
print(color)
intr = cam.get_intrinsics()
print(intr)
verts, tex = cam.get_pointcloud()
cam.stop()
'''
"""cam = RealSenseCamera(enable_filters=True)
color, depth = cam.get_frames()
for color, depth in cam.stream_frames():
    cv2.imshow('Color', color)
    depth_8u = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_8u = depth_8u.astype(np.uint8)
    cv2.imshow('Depth', depth_8u)
    if cv2.waitKey(1)==ord('q'): break
cam.stop()"""
if __name__ == '__main__':
    cam = RealSenseCamera(width=640, height=480, fps=30)
    with PointCloudVisualizer(cam) as viz:
        try:
            while True:
                viz.update()
        except KeyboardInterrupt:
            pass
    cam.stop()