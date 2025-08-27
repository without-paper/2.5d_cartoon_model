import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional
import math
from scipy.spatial import ConvexHull  # For point-in-poly check
import matplotlib.path as mpath  # For point-in-polygon
from svg.path import parse_path
import os
import re

class Camera:
    """相机类，表示观察视角，基于角色中心的球面坐标系"""
    def __init__(self, yaw: float = 0, pitch: float = 0, 
                 image_width: int = 210, image_height: int = 297, 
                 anchor_3d: Optional[np.ndarray] = None, radius: float = 10.0):
        self.yaw = yaw    # 逆时针旋转围绕垂直轴 (Y轴)，范围 [-π, π]
        self.pitch = pitch  # 顺时针旋转围绕水平轴 (X轴)，正值向下，范围 [-π/2, π/2]
        self.image_width = image_width
        self.image_height = image_height
        self.anchor_3d = np.array([0, 0, 0]) if anchor_3d is None else anchor_3d  # 角色3D中心
        self.radius = radius  # 相机到角色的距离

        # 计算相机位置 (球面坐标)
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # 相机位置 (X, Y, Z) 相对于anchor，Z向前，Y向上，X向右
        self.position = self.anchor_3d + np.array([
            self.radius * cos_pitch * sin_yaw,  # X
            self.radius * sin_pitch,            # Y
            -self.radius * cos_pitch * cos_yaw  # Z (负值因为Z向前)
        ])

        # 计算方向向量
        self.forward = -(self.position - self.anchor_3d) / self.radius  # 朝向anchor
        self.up = np.array([0, 1, 0])  # 初始向上向量
        self.right = np.cross(self.forward, self.up)  # 右向量
        self.up = np.cross(self.right, self.forward)  # 重新计算up确保正交
        self.up /= np.linalg.norm(self.up)  # 归一化
        self.right /= np.linalg.norm(self.right)  # 归一化

class Stroke:
    def __init__(self, stroke_id: str):
        self.id = stroke_id
        self.key_views: Dict[Tuple[float, float], np.ndarray] = {}
        self.anchor_3d: Optional[np.ndarray] = None
        self.has_anchor = True
        self.group_id: Optional[str] = None
        self.styles: Dict[str, str] = {}     # e.g., {'fill': '#ffcc00', 'stroke': '#000', 'stroke-width': '1'}
        self.areas: Dict[Tuple[float, float], float] = {}  # 记录每个视角下的多边形面积
        self.visibility_regions: List[List[Tuple[float, float]]] = []  # e.g., [[(yaw1,pitch1), (yaw2,pitch2), ...]]
        self.relative_offset: Optional[np.ndarray] = None

def _polygon_area(points: np.ndarray) -> float:
    if points is None or len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

class StrokeGroup:
    """表示一组相关的笔画"""
    def __init__(self, group_id: str):
        self.id = group_id
        self.strokes: List[Stroke] = []
        self.anchor_3d: Optional[np.ndarray] = None

class CartoonModel2_5D:
    """2.5D卡通模型"""
    def __init__(self, image_width: int = 210, image_height: int = 297):
        self.strokes: Dict[str, Stroke] = {}
        self.groups: Dict[str, StrokeGroup] = {}
        self.key_views: List[Tuple[float, float]] = []
        self.delaunay: Optional[Delaunay] = None
        self.image_width = image_width
        self.image_height = image_height
        self.view_orders: Dict[Tuple[float, float], List[str]] = {}
        # Default group anchor if none exists
        self.default_group_anchor = np.array([0, 0, 0])

    def add_key_view(self, camera: Camera, svg_data: Dict[str, np.ndarray], styles: Optional[Dict[str, Dict[str, str]]] = None, order: List[str] = None):
        """添加关键视角；svg_data 可来自 parse_svg_file 的第一返回值；styles 为第二返回值；order 为第三返回值"""
        view_key = (camera.yaw, camera.pitch)
        if view_key not in self.key_views:
            self.key_views.append(view_key)
        if order:
            self.view_orders[view_key] = order

        for stroke_id, path_points in svg_data.items():
            if stroke_id not in self.strokes:
                self.strokes[stroke_id] = Stroke(stroke_id)

            resampled_points = self._resample_path(path_points, 100)
            self.strokes[stroke_id].key_views[view_key] = resampled_points
            self.strokes[stroke_id].areas[view_key] = abs(_polygon_area(resampled_points))
            if styles and not self.strokes[stroke_id].styles:
                self.strokes[stroke_id].styles = styles.get(stroke_id, {})

    def center_views(self):
        """Align centroids across all views to image center."""
        global_centroids = {}
        for view in self.key_views:
            view_points = []
            for stroke in self.strokes.values():
                if view in stroke.key_views:
                    view_points.append(stroke.key_views[view])
            if view_points:
                all_points = np.vstack(view_points)
                global_centroids[view] = np.mean(all_points, axis=0)

        for view in self.key_views:
            if view in global_centroids:
                shift = np.array([self.image_width / 2, self.image_height / 2]) - global_centroids[view]
                for stroke in self.strokes.values():
                    if view in stroke.key_views:
                        stroke.key_views[view] += shift
                        stroke.areas[view] = abs(_polygon_area(stroke.key_views[view]))

    def _resample_path(self, points: np.ndarray, num_points: int) -> np.ndarray:
        """重采样路径到固定点数"""
        if len(points) <= 1:
            return points
            
        # 计算累积距离
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            distances[i] = distances[i-1] + np.linalg.norm(points[i] - points[i-1])
        
        if distances[-1] == 0:  # 所有点重合
            return np.tile(points[0], (num_points, 1))
        
        # 归一化距离
        normalized_distances = distances / distances[-1]
        
        # 创建插值函数
        interp_x = interp1d(normalized_distances, points[:, 0], kind='linear')
        interp_y = interp1d(normalized_distances, points[:, 1], kind='linear')
        
        # 在新距离上采样
        new_distances = np.linspace(0, 1, num_points)
        new_x = interp_x(new_distances)
        new_y = interp_y(new_distances)
        
        return np.column_stack([new_x, new_y])

    def compute_3d_anchors(self):
        """计算所有笔画的3D anchor点 with occlusion handling"""
        for stroke_id, stroke in self.strokes.items():
            if not stroke.has_anchor:
                continue
                
            centers_2d = []
            camera_poses = []
            
            for (yaw, pitch), path_points in stroke.key_views.items():
                # 计算2D中心点
                center_2d = np.mean(path_points, axis=0)
                centers_2d.append(center_2d)
                camera = Camera(yaw, pitch, self.image_width, self.image_height)
                camera_poses.append(camera)
            
            if len(centers_2d) >= 2:
                # Initial anchor estimate
                initial_anchor = np.zeros(3)
                for iteration in range(20):  # Maximum 20 iterations
                    projections = []
                    visible_count = 0
                    
                    for center_2d, camera in zip(centers_2d, camera_poses):
                        ray_origin, ray_dir = self._ray_from_2d_point(center_2d, camera)
                        t = np.dot(initial_anchor - ray_origin, ray_dir)
                        projection = ray_origin + t * ray_dir
                        # Check occlusion: if projection is behind another stroke's anchor, discard
                        occluded = False
                        for other_stroke in self.strokes.values():
                            if other_stroke.anchor_3d is not None and other_stroke != stroke:
                                dist_to_other = np.linalg.norm(projection - other_stroke.anchor_3d)
                                if dist_to_other < 1e-6:  # Simple occlusion check
                                    occluded = True
                                    break
                        if not occluded:
                            projections.append(projection)
                            visible_count += 1
                    
                    if visible_count < 2:  # Not enough visible projections
                        break
                    
                    new_anchor = np.mean(projections, axis=0)
                    if np.linalg.norm(new_anchor - initial_anchor) < 1e-6:
                        break
                    initial_anchor = new_anchor
                
                stroke.anchor_3d = initial_anchor
                print(f"计算 {stroke_id} 的3D anchor: {stroke.anchor_3d}")
    
    def _compute_single_anchor(self, centers_2d: List[np.ndarray], cameras: List[Camera]) -> np.ndarray:
        """计算单个笔画的3D anchor点"""
        anchor = np.zeros(3)
        
        for iteration in range(20):  # 最多迭代20次
            projections = []
            
            for center_2d, camera in zip(centers_2d, cameras):
                ray_origin, ray_dir = self._ray_from_2d_point(center_2d, camera)
                t = np.dot(anchor - ray_origin, ray_dir)
                projection = ray_origin + t * ray_dir
                projections.append(projection)
            
            new_anchor = np.mean(projections, axis=0)
            if np.linalg.norm(new_anchor - anchor) < 1e-6:
                break
            anchor = new_anchor
        
        return anchor
    
    def _ray_from_2d_point(self, point_2d: np.ndarray, camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
        w2 = camera.image_width / 2.0
        h2 = camera.image_height / 2.0
        rel_x = point_2d[0] - w2
        rel_y = point_2d[1] - h2
        ray_origin = camera.position + rel_x * camera.right + rel_y * camera.up
        ray_dir = camera.forward  # Direction toward anchor
        return ray_origin, ray_dir
    
    def build_delaunay_triangulation(self):
        """构建Delaunay三角剖分"""
        if len(self.key_views) >= 3:
            points = np.array(self.key_views)
            self.delaunay = Delaunay(points)
            print(f"构建Delaunay三角剖分，包含 {len(self.delaunay.simplices)} 个三角形")
    
    def _get_rotation_matrix(self, center: np.ndarray, delta_yaw: float, delta_pitch: float) -> np.ndarray:
        """Generate rotation matrix for global rotation around center"""
        cos_yaw = math.cos(delta_yaw)
        sin_yaw = math.sin(delta_yaw)
        cos_pitch = math.cos(delta_pitch)
        sin_pitch = math.sin(delta_pitch)
        
        # Rotation matrix for yaw (around Y-axis) then pitch (around X-axis)
        R_yaw = np.array([
            [cos_yaw, 0, sin_yaw],
            [0, 1, 0],
            [-sin_yaw, 0, cos_yaw]
        ])
        R_pitch = np.array([
            [1, 0, 0],
            [0, cos_pitch, -sin_pitch],
            [0, sin_pitch, cos_pitch]
        ])
        R = R_pitch @ R_yaw
        
        # Translate to center, rotate, translate back
        T = np.eye(4)
        T[:3, 3] = -center
        T_inv = np.eye(4)
        T_inv[:3, 3] = center
        R_4x4 = np.eye(4)
        R_4x4[:3, :3] = R
        return T_inv @ R_4x4 @ T

    def _find_closest_view(self, target_yaw: float, target_pitch: float) -> Tuple[float, float]:
        """找到最近的已知视角"""
        min_dist = float('inf')
        closest_view = self.key_views[0] if self.key_views else (0, 0)
        
        for yaw, pitch in self.key_views:
            dist = math.sqrt((yaw - target_yaw)**2 + (pitch - target_pitch)**2)
            if dist < min_dist:
                min_dist = dist
                closest_view = (yaw, pitch)
        
        return closest_view
    
    def interpolate_shape(self, stroke_id: str, target_camera: Camera) -> Optional[np.ndarray]:
        stroke = self.strokes[stroke_id]
        target_view = (target_camera.yaw, target_camera.pitch)

        if not self._is_visible(stroke, target_view):
            return None

        if target_view in stroke.key_views:
            return stroke.key_views[target_view]

        if self.delaunay is None or len(stroke.key_views) < 2:
            return self._interpolate_nearest_edge(stroke, target_view)

        target_point = np.array(target_view).reshape(1, -1)
        simplex_idx = self.delaunay.find_simplex(target_point)
        if simplex_idx[0] != -1:
            triangle_indices = self.delaunay.simplices[simplex_idx[0]]
            triangle_views = [tuple(self.delaunay.points[i]) for i in triangle_indices]
            available_map = {v: s for v, s in zip(triangle_views, [stroke.key_views.get(v) for v in triangle_views]) if s is not None}
            available_views = list(available_map.keys())
            available_shapes = list(available_map.values())
            available_areas = [stroke.areas.get(v, 0.0) for v in available_views]

            if len(available_views) < 2:
                return self._interpolate_nearest_edge(stroke, target_view)

            transform = self.delaunay.transform[simplex_idx[0]]
            bary = transform[:2].dot(target_point[0] - transform[2])
            bary = np.append(bary, 1 - bary.sum())  # [u, v, w]

            weight_map = {triangle_views[i]: bary[i] for i in range(3)}
            weights = np.array([weight_map[v] for v in available_views], dtype=float)
            weights = weights / weights.sum()

            min_points = min(len(shape) for shape in available_shapes)
            aligned_shapes = [self._resample_path(shape, min_points) if len(shape) != min_points else shape for shape in available_shapes]

            if len(available_shapes) >= 2:
                interpolated = self._advanced_interpolate(aligned_shapes, weights)
            else:
                interpolated = sum(w * s for w, s in zip(weights, aligned_shapes))

            area_targets = np.array([stroke.areas.get(v, 0.0) for v in available_views], dtype=float)
            area_target = float(np.dot(weights, area_targets))
            area_cur = abs(_polygon_area(interpolated))
            if area_cur > 1e-8 and area_target > 1e-8:
                scale = math.sqrt(area_target / area_cur)
                c = np.mean(interpolated, axis=0, keepdims=True)
                interpolated = (interpolated - c) * scale + c

            return interpolated
        else:
            return self._interpolate_nearest_edge(stroke, target_view)
    
    def _interpolate_nearest_edge(self, stroke: Stroke, target_view: Tuple[float, float]) -> Optional[np.ndarray]:
        """Project onto nearest triangle edge for interpolation."""
        target_point = np.array(target_view)
        min_dist = float('inf')
        closest_t = None
        closest_views = None

        for simplex in self.delaunay.simplices:
            verts = self.delaunay.points[simplex]
            for i in range(3):
                edge_start = verts[i]
                edge_end = verts[(i + 1) % 3]
                vec = edge_end - edge_start
                proj = np.dot(target_point - edge_start, vec) / np.dot(vec, vec)
                proj = np.clip(proj, 0, 1)
                proj_point = edge_start + proj * vec
                dist = np.linalg.norm(target_point - proj_point)
                if dist < min_dist:
                    min_dist = dist
                    closest_t = proj
                    closest_views = (tuple(edge_start), tuple(edge_end))

        if closest_views:
            v1, v2 = closest_views
            if v1 in stroke.key_views and v2 in stroke.key_views:
                shape1 = stroke.key_views[v1]
                shape2 = stroke.key_views[v2]
                min_points = min(len(shape1), len(shape2))
                shape1 = self._resample_path(shape1, min_points)
                shape2 = self._resample_path(shape2, min_points)
                interpolated = (1 - closest_t) * shape1 + closest_t * shape2
                area_target = (1 - closest_t) * stroke.areas.get(v1, 0) + closest_t * stroke.areas.get(v2, 0)
                area_cur = abs(_polygon_area(interpolated))
                if area_cur > 1e-8 and area_target > 0:
                    scale = math.sqrt(area_target / area_cur)
                    c = np.mean(interpolated, axis=0, keepdims=True)
                    interpolated = (interpolated - c) * scale + c
                return interpolated
        return None
    
    def _advanced_interpolate(self, shapes: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        """Advanced: Sederberg for correspondence + Alexa for ARAP paths."""
        if len(shapes) == 2:
            shape1, shape2 = shapes
            w1, w2 = weights
            corr = self._sederberg_correspondence(shape1, shape2)
            aligned1, aligned2 = self._align_to_correspondence(shape1, shape2, corr)
            interpolated = self._arap_interpolate(aligned1, aligned2, w2)
            return interpolated
        return sum(w * s for w, s in zip(weights, shapes))

    def _sederberg_correspondence(self, poly1: np.ndarray, poly2: np.ndarray) -> List[Tuple[int, int]]:
        """Sederberg DP for vertex correspondence (min work)."""
        m, n = len(poly1), len(poly2)
        k_s = 1.0
        e_s = 1.0
        c_s = 0.5
        k_b = 1.0
        e_b = 1.0
        m_b = 1.0
        p_b = 10.0

        cost = np.full((m + 1, n + 1), float('inf'))
        cost[0, 0] = 0
        backtrack = np.full((m + 1, n + 1, 2), -1, dtype=int)

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 and j == 0:
                    continue
                
                if i > 0:
                    insert_cost = k_s * np.linalg.norm(poly1[i-1]) ** e_s
                    new_cost = cost[i-1, j] + insert_cost
                    if new_cost < cost[i, j]:
                        cost[i, j] = new_cost
                        backtrack[i, j] = [i-1, j]
                
                if j > 0:
                    insert_cost = k_s * np.linalg.norm(poly2[j-1]) ** e_s
                    new_cost = cost[i, j-1] + insert_cost
                    if new_cost < cost[i, j]:
                        cost[i, j] = new_cost
                        backtrack[i, j] = [i, j-1]
                
                if i > 0 and j > 0:
                    L0 = np.linalg.norm(poly1[i-1]) if i==1 else np.linalg.norm(poly1[i-2] - poly1[i-1])
                    L1 = np.linalg.norm(poly2[j-1]) if j==1 else np.linalg.norm(poly2[j-2] - poly2[j-1])
                    stretch = k_s * abs(L1 - L0) ** e_s * ((1 - c_s) * min(L0, L1) + c_s * max(L0, L1))
                    
                    if i > 1 and j > 1:
                        vec1_prev = poly1[i-2] - poly1[i-1]
                        vec1_curr = poly1[i-1] - poly1[i]
                        theta0 = np.arccos(np.dot(vec1_prev, vec1_curr) / (np.linalg.norm(vec1_prev) * np.linalg.norm(vec1_curr)))
                        
                        vec2_prev = poly2[j-2] - poly2[j-1]
                        vec2_curr = poly2[j-1] - poly2[j]
                        theta1 = np.arccos(np.dot(vec2_prev, vec2_curr) / (np.linalg.norm(vec2_prev) * np.linalg.norm(vec2_curr)))
                        
                        d_theta = abs(theta1 - theta0)
                        d_theta_star = 0
                        if theta0 == 0 or theta1 == 0:
                            bend = k_b * (d_theta + m_b * d_theta_star) ** e_b + p_b
                        else:
                            bend = k_b * (d_theta + m_b * d_theta_star) ** e_b
                    else:
                        bend = 0
                    
                    new_cost = cost[i-1, j-1] + stretch + bend
                    if new_cost < cost[i, j]:
                        cost[i, j] = new_cost
                        backtrack[i, j] = [i-1, j-1]

        corr = []
        i, j = m, n
        while i > 0 or j > 0:
            prev = backtrack[i, j]
            if prev[0] == i-1 and prev[1] == j-1:
                corr.append((i-1, j-1))
            i, j = prev
        return corr[::-1]

    def _stretching_cost(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return abs(np.linalg.norm(p1) - np.linalg.norm(p2))

    def _insertion_cost(self, p: np.ndarray) -> float:
        return np.linalg.norm(p) * 0.5

    def _align_to_correspondence(self, poly1: np.ndarray, poly2: np.ndarray, corr: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        len1 = np.cumsum(np.linalg.norm(np.diff(poly1, axis=0), axis=1))
        len1 = np.insert(len1, 0, 0)
        len2 = np.cumsum(np.linalg.norm(np.diff(poly2, axis=0), axis=1))
        len2 = np.insert(len2, 0, 0)
        
        len1 /= len1[-1]
        len2 /= len2[-1]
        
        aligned1 = []
        aligned2 = []
        prev_i, prev_j = 0, 0
        for i, j in corr:
            seg1 = poly1[prev_i:i+1]
            seg2 = poly2[prev_j:j+1]
            num_pts = max(len(seg1), len(seg2))
            aligned1.append(self._resample_path(seg1, num_pts))
            aligned2.append(self._resample_path(seg2, num_pts))
            prev_i, prev_j = i, j
        
        aligned1 = np.vstack(aligned1) if aligned1 else poly1
        aligned2 = np.vstack(aligned2) if aligned2 else poly2
        return aligned1, aligned2

    def _arap_interpolate(self, poly1: np.ndarray, poly2: np.ndarray, t: float) -> np.ndarray:
        if not np.allclose(poly1[0], poly1[-1]):
            poly1 = np.vstack([poly1, poly1[0]])
            poly2 = np.vstack([poly2, poly2[0]])
        
        tri = Delaunay(poly1[:-1])
        triangles = tri.simplices
        
        V0 = poly1[:-1]
        V1 = poly2[:-1]
        n = len(V0)
        
        Vt = (1 - t) * V0 + t * V1
        for iter in range(10):
            R = []
            for idx in triangles:
                P = V0[idx]
                Q = Vt[idx]
                P_c = P - np.mean(P, axis=0)
                Q_c = Q - np.mean(Q, axis=0)
                C = Q_c.T @ P_c
                U, S, Vh = np.linalg.svd(C)
                Ri = U @ Vh
                if np.linalg.det(Ri) < 0:
                    U[:, -1] = -U[:, -1]
                    Ri = U @ Vh
                R.append(Ri)
            
            A = np.zeros((2*n, 2*n))
            b = np.zeros((2*n, 1))
            for tri_idx, idx in enumerate(triangles):
                Ri = R[tri_idx]
                for j in range(3):
                    vj = idx[j]
                    w = 1.0
                    A[2*vj:2*vj+2, 2*vj:2*vj+2] += w * np.eye(2)
                    for k in range(3):
                        if j == k: continue
                        vk = idx[k]
                        e_jk_0 = V0[vk] - V0[vj]
                        A[2*vj:2*vj+2, 2*vk:2*vk+2] -= (w / 2) * Ri
                        A[2*vj:2*vj+2, 2*vj:2*vj+2] += (w / 2) * Ri
                        e_jk_1 = V1[vk] - V1[vj]
                        b[2*vj:2*vj+2] += (w / 2) * Ri @ ((1 - t) * e_jk_0 + t * e_jk_1)
            
            x = np.linalg.solve(A, b.flatten())
            Vt_new = x.reshape(n, 2)
            if np.linalg.norm(Vt_new - Vt) < 1e-6:
                break
            Vt = Vt_new
        
        return Vt

    def _is_visible(self, stroke: Stroke, target_view: Tuple[float, float]) -> bool:
        """Check if target (yaw,pitch) inside any visibility polygon."""
        for poly in stroke.visibility_regions:
            if len(poly) < 3:
                continue
            path = mpath.Path(np.array(poly))
            if path.contains_point(target_view):
                return True
        return True
    
    def add_key_view(self, camera: Camera, svg_data: Dict[str, np.ndarray], styles: Optional[Dict[str, Dict[str, str]]] = None, order: List[str] = None):
        """添加关键视角；svg_data 可来自 parse_svg_file 的第一返回值；styles 为第二返回值；order 为第三返回值"""
        view_key = (camera.yaw, camera.pitch)
        if view_key not in self.key_views:
            self.key_views.append(view_key)
        if order:
            self.view_orders[view_key] = order  # Store the original SVG order

        for stroke_id, path_points in svg_data.items():
            if stroke_id not in self.strokes:
                self.strokes[stroke_id] = Stroke(stroke_id)

            resampled_points = self._resample_path(path_points, 100)
            self.strokes[stroke_id].key_views[view_key] = resampled_points
            self.strokes[stroke_id].areas[view_key] = abs(_polygon_area(resampled_points))
            if styles and not self.strokes[stroke_id].styles:
                self.strokes[stroke_id].styles = styles.get(stroke_id, {})

    def render_novel_view(self, target_camera: Camera) -> Dict[str, np.ndarray]:
        """渲染新视角：若命中已知关键视角，直接返回该形状；否则按anchor+插值 with global rotation and Z-ordering"""
        stroke_list = []
        target_view = (target_camera.yaw, target_camera.pitch)
        
        # Use head group anchor if exists, otherwise default
        if self.groups:
            group_anchor = self.groups.get('head', next(iter(self.groups.values()))).anchor_3d
        else:
            group_anchor = self.default_group_anchor
        
        print(f"Rendering with group_anchor: {group_anchor}")
        
        # Apply global rotation to head group anchor
        rotation_matrix = self._get_rotation_matrix(group_anchor, target_camera.yaw, target_camera.pitch)
        rotated_group_anchor_4d = rotation_matrix @ np.append(group_anchor, 1)
        rotated_group_anchor = rotated_group_anchor_4d[:3]
        
        # Get the base order for this view or the first available view
        base_order = self.view_orders.get(target_view)
        if not base_order and self.view_orders:
            base_order = next(iter(self.view_orders.values()))  # Fallback to first view's order
        
        for stroke_id, stroke in self.strokes.items():
            try:
                if not self._is_visible(stroke, target_view):
                    continue

                if target_view in stroke.key_views:
                    shape = stroke.key_views[target_view]
                else:
                    shape = self.interpolate_shape(stroke_id, target_camera)
                    if shape is None or len(shape) < 2:
                        continue
                    # Use hierarchical group-relative positioning
                    if stroke.group_id and stroke.group_id in self.groups:
                        current_group = self.groups[stroke.group_id]
                        if current_group.anchor_3d is not None and stroke.relative_offset is not None:
                            # Traverse up to head group for global rotation
                            group_anchor_to_use = current_group.anchor_3d
                            while current_group.group_id and current_group.group_id in self.groups:
                                parent_group = self.groups[current_group.group_id]
                                group_anchor_to_use = parent_group.anchor_3d
                                current_group = parent_group
                            rotated_anchor = rotated_group_anchor + stroke.relative_offset
                            print(f"Stroke {stroke_id} rotated_anchor: {rotated_anchor}")
                            pos_2d = self._project_3d_to_2d(rotated_anchor, target_camera)
                        else:
                            pos_2d = None
                    elif stroke.anchor_3d is not None:
                        anchor_4d = np.append(stroke.anchor_3d, 1)
                        rotated_anchor_4d = rotation_matrix @ anchor_4d
                        pos_2d = self._project_3d_to_2d(rotated_anchor_4d[:3], target_camera)
                    else:
                        pos_2d = None
                    if pos_2d is not None:
                        shape_center = np.mean(shape, axis=0)
                        shape = shape - shape_center + pos_2d

                # Determine Z-order: prioritize SVG order, use depth for novel views with valid anchors
                if base_order and stroke_id in base_order:
                    order_index = base_order.index(stroke_id)  # Lower index = lower layer
                else:
                    order_index = len(base_order or [])  # Default to end if not in order
                
                # Use anchor for depth calculation if available
                if hasattr(stroke, 'anchor_3d') and stroke.anchor_3d is not None:
                    depth = -np.dot(stroke.anchor_3d - target_camera.position, target_camera.forward)
                else:
                    depth = order_index * float('inf')

                stroke_list.append((stroke_id, shape, depth, order_index))

            except Exception as e:
                print(f"渲染笔画 {stroke_id} 时出错: {e}")
                continue

        # Sort primarily by original SVG order (order_index, ascending = bottom to top), then by depth for novel views
        stroke_list.sort(key=lambda x: (x[3], x[2] if target_view not in self.key_views else 0))
        ordered_rendered = {sid: shape for sid, shape, _, _ in stroke_list}
        return ordered_rendered
    
    def _get_rotation_matrix(self, center: np.ndarray, delta_yaw: float, delta_pitch: float) -> np.ndarray:
        """Generate rotation matrix for global rotation around center"""
        # Fallback to default if center is None
        if center is None:
            center = np.array([0, 0, 0])
        cos_yaw = math.cos(delta_yaw)
        sin_yaw = math.sin(delta_yaw)
        cos_pitch = math.cos(delta_pitch)
        sin_pitch = math.sin(delta_pitch)
        
        # Rotation matrix for yaw (around Y-axis) then pitch (around X-axis)
        R_yaw = np.array([
            [cos_yaw, 0, sin_yaw],
            [0, 1, 0],
            [-sin_yaw, 0, cos_yaw]
        ])
        R_pitch = np.array([
            [1, 0, 0],
            [0, cos_pitch, -sin_pitch],
            [0, sin_pitch, cos_pitch]
        ])
        R = R_pitch @ R_yaw
        
        # Translate to center, rotate, translate back
        T = np.eye(4)
        T[:3, 3] = -center
        T_inv = np.eye(4)
        T_inv[:3, 3] = center
        R_4x4 = np.eye(4)
        R_4x4[:3, :3] = R
        return T_inv @ R_4x4 @ T
    
    def _project_3d_to_2d(self, point_3d: np.ndarray, camera: Camera) -> Optional[np.ndarray]:
        direction = point_3d - camera.position
        x_2d = np.dot(direction, camera.right)
        y_2d = np.dot(direction, camera.up)
        x_image = x_2d + camera.image_width / 2
        y_image = y_2d + camera.image_height / 2  # Invert Y for SVG
        return np.array([x_image, y_image])
    
    def create_group(self, group_id: str, stroke_ids: List[str], parent_group_id: Optional[str] = None):
        """创建笔画组 with relative offsets, optionally under a parent group"""
        group = StrokeGroup(group_id)
        
        for stroke_id in stroke_ids:
            if stroke_id in self.strokes:
                stroke = self.strokes[stroke_id]
                stroke.group_id = group_id
                group.strokes.append(stroke)
        
        self.groups[group_id] = group
        if parent_group_id in self.groups:
            # Handle hierarchical grouping if needed
            pass

        anchors = []
        for stroke in group.strokes:
            if stroke.anchor_3d is not None:
                anchors.append(stroke.anchor_3d)
        
        if anchors:
            group.anchor_3d = np.mean(anchors, axis=0)
            print(f"创建组 {group_id}，anchor点: {group.anchor_3d}")
            # Store relative offsets for all strokes in the group
            for stroke in group.strokes:
                if stroke.anchor_3d is not None:
                    if stroke.relative_offset is None:
                        stroke.relative_offset = stroke.anchor_3d - group.anchor_3d
                    else:
                        stroke.relative_offset = stroke.anchor_3d - group.anchor_3d  # Update if already set
                else:
                    stroke.relative_offset = np.zeros(3)
        else:
            group.anchor_3d = np.zeros(3)  # Default anchor if no valid anchors
        
        self.groups[group_id] = group
        if parent_group_id and parent_group_id in self.groups:
            self.groups[parent_group_id].strokes.append(group)  # Hierarchical linking

def parse_svg_file(svg_file: str, view_prefix: str = "") -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, str]], List[str]]:
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    strokes = {}
    styles = {}
    order = []

    def _parse_float_with_units(value: str, default: float) -> float:
        """Convert string with units (e.g., '210mm') to float, ignoring units."""
        try:
            # Strip non-numeric characters after digits (e.g., 'mm', 'px')
            numeric = re.sub(r'[^\d.-]', '', value)
            return float(numeric)
        except (ValueError, TypeError):
            return default

    def _parse_transform(transform: str) -> np.ndarray:
        """Parse SVG transform attribute into a 3x3 transformation matrix."""
        matrix = np.eye(3)
        if not transform:
            return matrix
        transforms = re.findall(r'(translate|scale|rotate|matrix)\((.*?)\)', transform)
        for t_type, params in transforms:
            params = [float(p) for p in params.replace(',', ' ').split() if p]
            if t_type == 'translate':
                tx, ty = params if len(params) == 2 else (params[0], 0)
                t = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            elif t_type == 'scale':
                sx, sy = params if len(params) == 2 else (params[0], params[0])
                t = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
            elif t_type == 'rotate':
                a = np.radians(params[0])
                if len(params) == 3:
                    cx, cy = params[1], params[2]
                    t1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
                    t2 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
                    t3 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
                    t = t3 @ t2 @ t1
                else:
                    t = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            elif t_type == 'matrix':
                a, b, c, d, e, f = params
                t = np.array([[a, c, e], [b, d, f], [0, 0, 1]])
            matrix = t @ matrix
        return matrix

    def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply transformation matrix to points."""
        if points.size == 0:
            return points
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
        transformed = points_homogeneous @ transform.T
        return transformed[:, :2]

    def _get_style(elem, parent_default=None):
        style = {'fill': 'none', 'stroke': '#000000', 'stroke-width': '1'}
        if parent_default:
            style.update({k: v for k, v in parent_default.items() if v is not None})
        inline = elem.get('style')
        if inline:
            parts = [p.strip() for p in inline.split(';') if p.strip()]
            for p in parts:
                if ':' in p:
                    k, v = p.split(':', 1)
                    style[k.strip()] = v.strip()
        for k in ['fill', 'stroke', 'stroke-width', 'stroke-opacity', 'fill-opacity', 'fill-rule']:
            v = elem.get(k)
            if v is not None and v != '':
                style[k] = v
        return style

    def _sample_path(d: str, samples_per_seg: int = 32) -> List[List[float]]:
        pts = []
        parsed = parse_path(d) if d else None
        if parsed:
            for seg in parsed:
                for t in np.linspace(0.0, 1.0, samples_per_seg, endpoint=True):
                    z = seg.point(t)
                    pts.append([z.real, z.imag])
        return pts

    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()

        # Parse SVG dimensions and viewBox
        width = _parse_float_with_units(root.get('width', '210'), 210)
        height = _parse_float_with_units(root.get('height', '297'), 297)
        viewBox = root.get('viewBox')
        viewbox_matrix = np.eye(3)
        if viewBox:
            vb_x, vb_y, vb_w, vb_h = map(lambda x: _parse_float_with_units(x, 0), viewBox.split())
            scale_x = 210 / vb_w if vb_w != 0 else 1
            scale_y = 297 / vb_h if vb_h != 0 else 1
            translate_x = -vb_x * scale_x
            translate_y = -vb_y * scale_y
            viewbox_matrix = np.array([
                [scale_x, 0, translate_x],
                [0, scale_y, translate_y],
                [0, 0, 1]
            ])

        # Process groups and elements
        for elem in root.iter():
            if elem.tag.endswith('g'):
                g_id = elem.get('id', f'group_{len(order)}')
                g_transform = _parse_transform(elem.get('transform'))
                g_style = _get_style(elem)
                for sub_elem in elem.iter():
                    if sub_elem.tag.endswith('path'):
                        stroke_id = sub_elem.get('id', f"{view_prefix}{g_id}_path_{len(strokes)}")
                        d = sub_elem.get('d', '')
                        points = _sample_path(d)
                        if points:
                            points = np.array(points, dtype=float)
                            transform = _parse_transform(sub_elem.get('transform')) @ g_transform @ viewbox_matrix
                            points = _apply_transform(points, transform)
                            strokes[stroke_id] = points
                            styles[stroke_id] = _get_style(sub_elem, g_style)
                            styles[stroke_id]['original_d'] = d  # Store original path for export
                            order.append(stroke_id)
                    elif sub_elem.tag.endswith('ellipse'):
                        stroke_id = sub_elem.get('id', f"{view_prefix}{g_id}_ellipse_{len(strokes)}")
                        try:
                            cx = _parse_float_with_units(sub_elem.get('cx', '0'), 0)
                            cy = _parse_float_with_units(sub_elem.get('cy', '0'), 0)
                            rx = _parse_float_with_units(sub_elem.get('rx', '0'), 0)
                            ry = _parse_float_with_units(sub_elem.get('ry', '0'), 0)
                            points = []
                            for angle in np.linspace(0, 2*np.pi, 64, endpoint=True):
                                x = cx + rx * np.cos(angle)
                                y = cy + ry * np.sin(angle)
                                points.append([x, y])
                            points = np.array(points, dtype=float)
                            transform = _parse_transform(sub_elem.get('transform')) @ g_transform @ viewbox_matrix
                            points = _apply_transform(points, transform)
                            strokes[stroke_id] = points
                            styles[stroke_id] = _get_style(sub_elem, g_style)
                            order.append(stroke_id)
                        except ValueError:
                            continue
                    elif sub_elem.tag.endswith('circle'):
                        stroke_id = sub_elem.get('id', f"{view_prefix}{g_id}_circle_{len(strokes)}")
                        try:
                            cx = _parse_float_with_units(sub_elem.get('cx', '0'), 0)
                            cy = _parse_float_with_units(sub_elem.get('cy', '0'), 0)
                            r = _parse_float_with_units(sub_elem.get('r', '0'), 0)
                            points = []
                            for angle in np.linspace(0, 2*np.pi, 64, endpoint=True):
                                x = cx + r * np.cos(angle)
                                y = cy + r * np.sin(angle)
                                points.append([x, y])
                            points = np.array(points, dtype=float)
                            transform = _parse_transform(sub_elem.get('transform')) @ g_transform @ viewbox_matrix
                            points = _apply_transform(points, transform)
                            strokes[stroke_id] = points
                            styles[stroke_id] = _get_style(sub_elem, g_style)
                            order.append(stroke_id)
                        except ValueError:
                            continue

        print(f"Parsed {len(strokes)} strokes from {svg_file} with order {order}")
        for sid, points in strokes.items():
            print(f"  Stroke {sid}: {len(points)} points, first point: {points[0] if len(points) > 0 else 'empty'}")
        return strokes, styles, order

    except FileNotFoundError:
        print(f"SVG file not found: {svg_file}")
        return {}, {}, []
    except ET.ParseError as e:
        print(f"XML parsing error in {svg_file}: {e}")
        return {}, {}, []
    except Exception as e:
        print(f"Error parsing SVG file {svg_file}: {e}")
        return {}, {}, []

def export_svg(strokes: Dict[str, np.ndarray], styles: Dict[str, Dict[str, str]], width: int, height: int, filename: str, order: List[str] = None):
    def _path_d(points: np.ndarray, close_path: bool = True) -> str:
        if len(points) == 0:
            return ""
        cmds = [f"M {points[0,0]:.3f},{points[0,1]:.3f}"]
        for i in range(1, len(points)):
            cmds.append(f"L {points[i,0]:.3f},{points[i,1]:.3f}")
        if close_path:
            cmds.append("Z")
        return " ".join(cmds)

    root = ET.Element("svg", attrib={
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(width),
        "height": str(height),
        "viewBox": f"0 0 {width} {height}"
    })

    stroke_ids = order if order else strokes.keys()
    for sid in stroke_ids:
        if sid in strokes:
            g = ET.SubElement(root, "g", attrib={"id": sid})
            style = styles.get(sid, {})
            path_d = style.get('original_d', _path_d(strokes[sid], close_path=style.get('fill', 'none') != 'none'))
            path_attr = {
                "d": path_d,
                "fill": style.get("fill", "none"),
                "stroke": style.get("stroke", "#000"),
                "stroke-width": style.get("stroke-width", "1")
            }
            for k in ['stroke-opacity', 'fill-opacity', 'fill-rule']:
                if k in style:
                    path_attr[k] = style[k]
            ET.SubElement(g, "path", attrib=path_attr)

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ", level=0)
    except AttributeError:
        pass

    tree.write(filename, encoding="utf-8", xml_declaration=True)
    print(f"SVG exported: {filename}")

def visualize_strokes(strokes: Dict[str, np.ndarray], output_file: str = None):
    if not strokes:
        print("没有笔画可可视化")
        return
        
    fig, ax = plt.subplots(figsize=(12, 12))
    
    colors = plt.cm.tab10.colors
    color_idx = 0
    
    for stroke_id, points in strokes.items():
        if len(points) > 1:
            path = Path(points)
            patch = patches.PathPatch(path, facecolor='none', 
                                    edgecolor=colors[color_idx % len(colors)], 
                                    lw=2, label=stroke_id)
            ax.add_patch(patch)
            color_idx += 1
            center = np.mean(points, axis=0)
            ax.plot(center[0], center[1], 'ro', markersize=5)
    
    ax.set_xlim(0, 210)
    ax.set_ylim(0, 297)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.legend()
    ax.set_title('2.5D Cartoon Model - Novel View')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {output_file}")
    else:
        plt.show()
    
    plt.close()

def generate_12_view_orientation_space(model: CartoonModel2_5D):
    print("Generating proper 12-view Parameterized Orientation Space...")
    
    original_views = [(0, 0), (0, math.pi/2), (-math.pi/2, 0)]  # front, top, right
    mirrored_views = []
    
    # Mirror views
    if (0, 0) in model.key_views:
        mirrored_views.append((math.pi, 0))  # back view
    if (-math.pi/2, 0) in model.key_views:
        mirrored_views.append((math.pi/2, 0))  # left view
    if (0, math.pi/2) in model.key_views:
        mirrored_views.append((0, -math.pi/2))  # bottom view
    
    # Add mirrored views
    for view in mirrored_views:
        if view not in model.key_views:
            model.key_views.append(view)
            source_view = None
            if view == (math.pi, 0):
                source_view = (0, 0)
            elif view == (math.pi/2, 0):
                source_view = (-math.pi/2, 0)
            elif view == (0, -math.pi/2):
                source_view = (0, math.pi/2)
            
            if source_view:
                # Set view order: inverse for back and bottom
                if view == (math.pi, 0):  # back
                    model.view_orders[view] = list(reversed(model.view_orders.get(source_view, [])))
                elif view[1] == -math.pi/2:  # all bottom views
                    model.view_orders[view] = list(reversed(model.view_orders.get(source_view, [])))
                else:
                    model.view_orders[view] = model.view_orders.get(source_view, [])
                
                for stroke_id, stroke in model.strokes.items():
                    if source_view in stroke.key_views:
                        original_shape = stroke.key_views[source_view]
                        mirrored_shape = original_shape.copy()
                        center_x = model.image_width / 2
                        center_y = model.image_height / 2
                        if view in [(math.pi, 0), (math.pi/2, 0)]:
                            mirrored_shape[:, 0] = 2 * center_x - mirrored_shape[:, 0]
                        elif view == (0, -math.pi/2):
                            mirrored_shape[:, 1] = 2 * center_y - mirrored_shape[:, 1]
                        stroke.key_views[view] = mirrored_shape
                        stroke.areas[view] = abs(_polygon_area(mirrored_shape))
                        print(f"Mirrored stroke {stroke_id} from {source_view} to {view}")
    
    print(f"Generated {len(mirrored_views)} mirrored views: {mirrored_views}")
    
    # Rotate top and bottom views
    rotated_views = []
    additional_yaws = [math.pi/2, -math.pi/2, math.pi]  # 90, -90, 180
    
    if (0, math.pi/2) in model.key_views:
        for yaw in additional_yaws:
            if (yaw, math.pi/2) not in model.key_views:
                rotated_views.append((yaw, math.pi/2))
    
    if (0, -math.pi/2) in model.key_views:
        for yaw in additional_yaws:
            if (yaw, -math.pi/2) not in model.key_views:
                rotated_views.append((yaw, -math.pi/2))
    
    for view in rotated_views:
        if view not in model.key_views:
            model.key_views.append(view)
            yaw, pitch = view
            source_view = (0, math.pi/2) if pitch == math.pi/2 else (0, -math.pi/2)
            
            # Set view order: same for top rotated, inverse for bottom rotated
            top_order = model.view_orders.get((0, math.pi/2), [])
            if pitch == math.pi/2:
                model.view_orders[view] = top_order
            else:
                model.view_orders[view] = list(reversed(top_order))
            
            for stroke_id, stroke in model.strokes.items():
                if source_view in stroke.key_views:
                    original_shape = stroke.key_views[source_view]
                    center = np.array([model.image_width / 2, model.image_height / 2])
                    rotation_angle = yaw
                    rotated_shape = original_shape.copy()
                    rotated_shape -= center
                    cos_a = math.cos(rotation_angle)
                    sin_a = math.sin(rotation_angle)
                    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    rotated_shape = np.dot(rotated_shape, rotation_matrix.T)
                    rotated_shape += center
                    stroke.key_views[view] = rotated_shape
                    stroke.areas[view] = abs(_polygon_area(rotated_shape))
                    print(f"Rotated stroke {stroke_id} from {source_view} to {view}")
    
    print(f"Generated {len(rotated_views)} rotated views: {rotated_views}")
    print(f"Total views in orientation space: {len(model.key_views)}")
    
    model.build_delaunay_triangulation()
    return model

def visualize_orientation_space_with_characters(model: CartoonModel2_5D, output_dir: str = "orientation_space_views"):
    """Visualize the Parameterized Orientation Space with character renderings"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a grid of subplots for the orientation space overview
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all views in the orientation space
    yaws = [view[0] for view in model.key_views]
    pitches = [view[1] for view in model.key_views]
    
    # Color code different types of views
    colors = []
    view_types = []
    
    for view in model.key_views:
        if view in [(0, 0), (0, math.pi/2), (-math.pi/2, 0)]:  # Original input views
            colors.append('red')
            view_types.append('Input')
        elif view in [(math.pi, 0), (math.pi/2, 0), (0, -math.pi/2)]:  # Mirrored views
            colors.append('green')
            view_types.append('Mirrored')
        else:  # Rotated views
            colors.append('blue')
            view_types.append('Rotated')
    
    # Create scatter plot
    scatter = ax.scatter(yaws, pitches, c=colors, s=100, alpha=0.7)
    
    # Add labels for important views
    important_views = {
        (0, 0): 'Front',
        (math.pi, 0): 'Back',
        (math.pi/2, 0): 'Left',
        (-math.pi/2, 0): 'Right',
        (0, math.pi/2): 'Top',
        (0, -math.pi/2): 'Bottom',
        (math.pi/2, math.pi/2): 'Top-Right',
        (-math.pi/2, math.pi/2): 'Top-Left',
        (math.pi, math.pi/2): 'Top-Back',
        (math.pi/2, -math.pi/2): 'Bottom-Right',
        (-math.pi/2, -math.pi/2): 'Bottom-Left',
        (math.pi, -math.pi/2): 'Bottom-Front'
    }
    
    for view, label in important_views.items():
        if view in model.key_views:
            idx = model.key_views.index(view)
            ax.annotate(label, (yaws[idx], pitches[idx]), xytext=(5, 5), 
                       textcoords='offset points', fontweight='bold', fontsize=8)
    
    # Set axis labels and title
    ax.set_xlabel('Yaw (radians)')
    ax.set_ylabel('Pitch (radians)')
    ax.set_title('Parameterized Orientation Space (12 Views)')
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Input Views'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Mirrored Views'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Rotated Views')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-math.pi/2, math.pi/2)
    
    # Save the orientation space overview
    overview_path = os.path.join(output_dir, "orientation_space_overview.png")
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Orientation space overview saved to {overview_path}")
    
    # Now create individual visualizations for the 12 main views
    print("Creating individual view visualizations for the 12 main views...")
    
    # Define the 12 main views we want to visualize
    main_views = [
        # Original input views
        (0, 0),           # Front
        (0, math.pi/2),   # Top
        (-math.pi/2, 0),  # Right
        
        # Mirrored views
        (math.pi, 0),     # Back
        (math.pi/2, 0),   # Left
        (0, -math.pi/2),  # Bottom
        
        # Rotated top views
        (math.pi/2, math.pi/2),    # Top-Right
        (-math.pi/2, math.pi/2),   # Top-Left
        (math.pi, math.pi/2),      # Top-Back
        
        # Rotated bottom views
        (math.pi/2, -math.pi/2),    # Bottom-Right
        (-math.pi/2, -math.pi/2),   # Bottom-Left
        (math.pi, -math.pi/2),      # Bottom-Front
    ]
    
    # Filter to only include views that exist in our model
    main_views = [view for view in main_views if view in model.key_views]
    
    # Create a figure with all main views in a grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, (yaw, pitch) in enumerate(main_views[:12]):  # Show first 12 main views
        ax = axes[i]
        
        # For key views, use the stored shapes directly
        view_strokes = {}
        for stroke_id, stroke in model.strokes.items():
            if (yaw, pitch) in stroke.key_views:
                view_strokes[stroke_id] = stroke.key_views[(yaw, pitch)]
        print(f"Rendering view (yaw={yaw:.2f}, pitch={pitch:.2f}) with {len(view_strokes)} strokes")
        
        # Get the base order for this view
        base_order = model.view_orders.get((yaw, pitch), [])
        
        # Order the stroke_ids according to base_order
        ordered_ids = [sid for sid in base_order if sid in view_strokes]
        missing = [sid for sid in view_strokes if sid not in ordered_ids]
        ordered_ids += sorted(missing)  # Append missing in sorted order
        
        # Plot the strokes in order
        colors = plt.cm.tab10.colors
        color_idx = 0
        
        for sid in ordered_ids:
            points = view_strokes[sid]
            if len(points) > 1:
                path = Path(points)
                patch = patches.PathPatch(path, facecolor='none', 
                                        edgecolor=colors[color_idx % len(colors)], 
                                        lw=2)
                ax.add_patch(patch)
                color_idx += 1
        
        ax.set_xlim(0, model.image_width)
        ax.set_ylim(0, model.image_height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Add title with view information
        view_name = f"Yaw: {math.degrees(yaw):.0f}°, Pitch: {math.degrees(pitch):.0f}°"
        
        # Check if this is a special view
        if (yaw, pitch) in important_views:
            view_name = f"{important_views[(yaw, pitch)]}\n{view_name}"
        
        ax.set_title(view_name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide any unused subplots
    for i in range(len(main_views[:12]), len(axes)):
        axes[i].set_visible(False)
    
    # Save the grid of views
    grid_path = os.path.join(output_dir, "12_main_views_grid.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"12 main views grid saved to {grid_path}")
    
    # Also save individual view images for the 12 main views
    for i, (yaw, pitch) in enumerate(main_views[:12]):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # For key views, use the stored shapes directly
        view_strokes = {}
        for stroke_id, stroke in model.strokes.items():
            if (yaw, pitch) in stroke.key_views:
                view_strokes[stroke_id] = stroke.key_views[(yaw, pitch)]
        
        # Get the base order for this view
        base_order = model.view_orders.get((yaw, pitch), [])
        
        # Order the stroke_ids according to base_order
        ordered_ids = [sid for sid in base_order if sid in view_strokes]
        missing = [sid for sid in view_strokes if sid not in ordered_ids]
        ordered_ids += sorted(missing)  # Append missing in sorted order
        
        # Plot the strokes in order
        colors = plt.cm.tab10.colors
        color_idx = 0
        
        for sid in ordered_ids:
            points = view_strokes[sid]
            if len(points) > 1:
                path = Path(points)
                patch = patches.PathPatch(path, facecolor='none', 
                                        edgecolor=colors[color_idx % len(colors)], 
                                        lw=2)
                ax.add_patch(patch)
                color_idx += 1
        
        ax.set_xlim(0, model.image_width)
        ax.set_ylim(0, model.image_height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Add title with view information
        view_name = f"Yaw: {math.degrees(yaw):.0f}°, Pitch: {math.degrees(pitch):.0f}°"
        if (yaw, pitch) in important_views:
            view_name = f"{important_views[(yaw, pitch)]} - {view_name}"
        
        ax.set_title(view_name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save individual view
        view_filename = f"view_{i:02d}_{important_views.get((yaw, pitch), 'unknown').lower().replace(' ', '_')}.png"
        view_path = os.path.join(output_dir, view_filename)
        plt.savefig(view_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"All individual views saved to {output_dir}")
    
    # 特别检查三个原始视图是否正确显示
    print("\n=== 检查原始输入视图 ===")
    original_views_to_check = [(0, 0), (0, math.pi/2), (-math.pi/2, 0)]
    original_view_names = ["Front", "Top", "Right"]
    
    for (yaw, pitch), name in zip(original_views_to_check, original_view_names):
        if (yaw, pitch) in model.key_views:
            # 计算这个视图中的笔画数量
            stroke_count = 0
            for stroke_id, stroke in model.strokes.items():
                if (yaw, pitch) in stroke.key_views:
                    stroke_count += 1
            
            print(f"{name} view (yaw={math.degrees(yaw):.0f}°, pitch={math.degrees(pitch):.0f}°): {stroke_count} strokes")
            
            # 检查是否有笔画数据
            if stroke_count == 0:
                print(f"  WARNING: No strokes found in {name} view!")
        else:
            print(f"  ERROR: {name} view not found in key views!")
    
    return output_dir

def main():
    model = CartoonModel2_5D(image_width=210, image_height=297)
    
    front_camera = Camera(yaw=0, pitch=0)
    top_camera = Camera(yaw=0, pitch=math.pi/2)
    right_camera = Camera(yaw=-math.pi/2, pitch=0)
    
    front_strokes, front_styles, front_order = parse_svg_file('data/yellow_head_front.svg', 'front_')
    top_strokes, top_styles, top_order = parse_svg_file('data/yellow_head_top.svg', 'top_')
    right_strokes, right_styles, right_order = parse_svg_file('data/yellow_head_right.svg', 'right_')
    
    print(f"Front strokes: {len(front_strokes)}, Order: {front_order}")
    print(f"Top strokes: {len(top_strokes)}, Order: {top_order}")
    print(f"Right strokes: {len(right_strokes)}, Order: {right_order}")
    
    # Export debug SVGs with order
    export_svg(front_strokes, front_styles, 210, 297, "debug_front_view.svg", front_order)
    export_svg(top_strokes, top_styles, 210, 297, "debug_top_view.svg", top_order)
    export_svg(right_strokes, right_styles, 210, 297, "debug_right_view.svg", right_order)
    
    model.add_key_view(front_camera, front_strokes, front_styles, front_order)
    model.add_key_view(top_camera, top_strokes, top_styles, top_order)
    model.add_key_view(right_camera, right_strokes, right_styles, right_order)
    
    print("Key views:", model.key_views)
    for stroke_id, stroke in model.strokes.items():
        print(f"Stroke {stroke_id} views:", list(stroke.key_views.keys()))
    
    model.center_views()
    model.compute_3d_anchors()
    model = generate_12_view_orientation_space(model)
    
    output_dir = visualize_orientation_space_with_characters(model, "orientation_space_views")
    
    model.create_group('eyes', ['front_rightEye', 'front_leftEye', 'top_rightEye', 'top_leftEye', 'right_rightEye', 'right_leftEye'])
    model.create_group('earsFace', ['front_rightEar', 'front_leftEar', 'front_face', 'top_rightEar', 'top_leftEar', 'top_face', 'right_rightEar', 'right_leftEar', 'right_face'], 'head')
    model.create_group('head', ['eyes', 'earsFace', 'front_nose', 'front_mouth', 'top_nose', 'top_mouth', 'right_nose', 'right_mouth'])
    
    # Note: render_novel_view is missing, commenting out
    novel_camera = Camera(yaw=math.pi/6, pitch=math.pi/6)
    rendered = model.render_novel_view(novel_camera)
    merged_styles = {}
    for d in [right_styles, top_styles, front_styles]:
        merged_styles.update(d)
    export_svg(rendered, merged_styles, model.image_width, model.image_height, "novel_view.svg")
    
    print("2.5D model created successfully!")
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()