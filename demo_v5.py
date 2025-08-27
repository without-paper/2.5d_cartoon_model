import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional
import math
from scipy.spatial import ConvexHull
import matplotlib.path as mpath
from svg.path import parse_path
import os
import re
import uuid

class Camera:
    """Camera class representing the viewpoint in spherical coordinates around the character's center."""
    def __init__(self, yaw: float = 0, pitch: float = 0, 
                 image_width: int = 210, image_height: int = 297, 
                 anchor_3d: Optional[np.ndarray] = None, radius: float = 10.0):
        self.yaw = yaw    # Counterclockwise rotation around vertical (Y) axis, range [-π, π]
        self.pitch = pitch  # Clockwise rotation around horizontal (X) axis, positive downward, range [-π/2, π/2]
        self.image_width = image_width
        self.image_height = image_height
        self.anchor_3d = np.array([0, 0, 0]) if anchor_3d is None else anchor_3d  # Character's 3D center
        self.radius = radius  # Camera distance to character

        # Compute camera position (spherical coordinates)
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # Camera position (X, Y, Z) relative to anchor, Z forward, Y up, X right
        self.position = self.anchor_3d + np.array([
            self.radius * cos_pitch * sin_yaw,  # X
            self.radius * sin_pitch,            # Y
            -self.radius * cos_pitch * cos_yaw  # Z (negative as Z is forward)
        ])

        # Compute direction vectors
        self.forward = -(self.position - self.anchor_3d) / self.radius  # Toward anchor
        self.up = np.array([0, 1, 0])  # Initial up vector
        self.right = np.cross(self.forward, self.up)  # Right vector
        self.up = np.cross(self.right, self.forward)  # Recalculate up for orthogonality
        self.up /= np.linalg.norm(self.up)  # Normalize
        self.right /= np.linalg.norm(self.right)  # Normalize

class Stroke:
    def __init__(self, stroke_id: str):
        self.id = stroke_id
        self.key_views: Dict[Tuple[float, float], np.ndarray] = {}
        self.anchor_3d: Optional[np.ndarray] = None
        self.has_anchor = True
        self.group_id: Optional[str] = None
        self.styles: Dict[str, str] = {}  # e.g., {'fill': '#ffcc00', 'stroke': '#000', 'stroke-width': '1'}
        self.areas: Dict[Tuple[float, float], float] = {}  # Polygon area per view
        self.visibility_regions: List[List[Tuple[float, float]]] = []  # e.g., [[(yaw1,pitch1), (yaw2,pitch2), ...]]
        self.relative_offset: Optional[np.ndarray] = None

def _polygon_area(points: np.ndarray) -> float:
    if points is None or len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

class StrokeGroup:
    """Represents a group of related strokes."""
    def __init__(self, group_id: str):
        self.id = group_id
        self.strokes: List[Stroke] = []
        self.anchor_3d: Optional[np.ndarray] = None

class CartoonModel2_5D:
    """2.5D cartoon model."""
    def __init__(self, image_width: int = 210, image_height: int = 297):
        self.strokes: Dict[str, Stroke] = {}
        self.groups: Dict[str, StrokeGroup] = {}
        self.key_views: List[Tuple[float, float]] = []
        self.delaunay: Optional[Delaunay] = None
        self.image_width = image_width
        self.image_height = image_height
        self.view_orders: Dict[Tuple[float, float], List[str]] = {}
        self.default_group_anchor = np.array([0, 0, 0])

    def add_key_view(self, camera: Camera, svg_data: Dict[str, np.ndarray], styles: Optional[Dict[str, Dict[str, str]]] = None, order: List[str] = None):
        """Add a key view; svg_data from parse_svg_file's first return; styles from second; order from third."""
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
        """Resample path to fixed number of points."""
        if len(points) <= 1:
            return points
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            distances[i] = distances[i-1] + np.linalg.norm(points[i] - points[i-1])
        if distances[-1] == 0:
            return np.tile(points[0], (num_points, 1))
        normalized_distances = distances / distances[-1]
        interp_x = interp1d(normalized_distances, points[:, 0], kind='linear')
        interp_y = interp1d(normalized_distances, points[:, 1], kind='linear')
        new_distances = np.linspace(0, 1, num_points)
        new_x = interp_x(new_distances)
        new_y = interp_y(new_distances)
        return np.column_stack([new_x, new_y])

    def compute_3d_anchors(self):
        """Compute 3D anchor points for all strokes with occlusion handling."""
        for stroke_id, stroke in self.strokes.items():
            if not stroke.has_anchor:
                continue
            centers_2d = []
            camera_poses = []
            for (yaw, pitch), path_points in stroke.key_views.items():
                center_2d = np.mean(path_points, axis=0)
                centers_2d.append(center_2d)
                camera = Camera(yaw, pitch, self.image_width, self.image_height)
                camera_poses.append(camera)
            if len(centers_2d) >= 2:
                initial_anchor = np.zeros(3)
                for iteration in range(20):
                    projections = []
                    visible_count = 0
                    for center_2d, camera in zip(centers_2d, camera_poses):
                        ray_origin, ray_dir = self._ray_from_2d_point(center_2d, camera)
                        t = np.dot(initial_anchor - ray_origin, ray_dir)
                        projection = ray_origin + t * ray_dir
                        occluded = False
                        for other_stroke in self.strokes.values():
                            if other_stroke.anchor_3d is not None and other_stroke != stroke:
                                dist_to_other = np.linalg.norm(projection - other_stroke.anchor_3d)
                                if dist_to_other < 1e-6:
                                    occluded = True
                                    break
                        if not occluded:
                            projections.append(projection)
                            visible_count += 1
                    if visible_count < 2:
                        break
                    new_anchor = np.mean(projections, axis=0)
                    if np.linalg.norm(new_anchor - initial_anchor) < 1e-6:
                        break
                    initial_anchor = new_anchor
                stroke.anchor_3d = initial_anchor
                print(f"Computed 3D anchor for {stroke_id}: {stroke.anchor_3d}")

    def _ray_from_2d_point(self, point_2d: np.ndarray, camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
        w2 = camera.image_width / 2.0
        h2 = camera.image_height / 2.0
        rel_x = point_2d[0] - w2
        rel_y = point_2d[1] - h2
        ray_origin = camera.position + rel_x * camera.right + rel_y * camera.up
        ray_dir = camera.forward
        return ray_origin, ray_dir

    def build_delaunay_triangulation(self):
        """Build Delaunay triangulation of key views."""
        if len(self.key_views) >= 3:
            points = np.array(self.key_views)
            self.delaunay = Delaunay(points)
            print(f"Built Delaunay triangulation with {len(self.delaunay.simplices)} triangles")

    def _get_rotation_matrix(self, center: np.ndarray, delta_yaw: float, delta_pitch: float) -> np.ndarray:
        """Generate rotation matrix for global rotation around center."""
        cos_yaw = math.cos(delta_yaw)
        sin_yaw = math.sin(delta_yaw)
        cos_pitch = math.cos(delta_pitch)
        sin_pitch = math.sin(delta_pitch)
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
        T = np.eye(4)
        T[:3, 3] = -center
        T_inv = np.eye(4)
        T_inv[:3, 3] = center
        R_4x4 = np.eye(4)
        R_4x4[:3, :3] = R
        return T_inv @ R_4x4 @ T

    def _find_closest_view(self, target_yaw: float, target_pitch: float) -> Tuple[float, float]:
        """Find the closest known view."""
        min_dist = float('inf')
        closest_view = self.key_views[0] if self.key_views else (0, 0)
        for yaw, pitch in self.key_views:
            dist = math.sqrt((yaw - target_yaw)**2 + (pitch - target_pitch)**2)
            if dist < min_dist:
                min_dist = dist
                closest_view = (yaw, pitch)
        return closest_view

    def _is_visible(self, stroke: Stroke, target_view: Tuple[float, float]) -> bool:
        """Check if stroke is visible in target view (placeholder, assumes always visible)."""
        return True  # Simplified for now; add visibility regions logic if needed

    def _advanced_interpolate(self, shapes: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        """Perform weighted interpolation of shapes (assumes aligned points)."""
        return np.average(shapes, axis=0, weights=weights)

    def _interpolate_nearest_edge(self, stroke: Stroke, target_view: Tuple[float, float]) -> Optional[np.ndarray]:
        """Interpolate shape from nearest edge or view."""
        min_dist = float('inf')
        best_shape = None
        for view in stroke.key_views:
            dist = math.sqrt((view[0] - target_view[0])**2 + (view[1] - target_view[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_shape = stroke.key_views[view]
        return best_shape

    def interpolate_shape(self, stroke_id: str, target_camera: Camera) -> Optional[np.ndarray]:
        """Interpolate stroke shape for target camera view."""
        stroke = self.strokes.get(stroke_id)
        if not stroke:
            return None
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
            if len(available_views) == 0:
                return self._interpolate_nearest_edge(stroke, target_view)
            elif len(available_views) == 1:
                return available_shapes[0]
            transform = self.delaunay.transform[simplex_idx[0]]
            bary = transform[:2].dot(target_point[0] - transform[2])
            bary = np.append(bary, 1 - bary.sum())
            weight_map = {triangle_views[i]: bary[i] for i in range(3)}
            weights = np.array([weight_map[v] for v in available_views], dtype=float)
            weights = weights / weights.sum()
            min_points = min(len(shape) for shape in available_shapes)
            aligned_shapes = [self._resample_path(shape, min_points) if len(shape) != min_points else shape for shape in available_shapes]
            interpolated = self._advanced_interpolate(aligned_shapes, weights)
            area_targets = np.array([stroke.areas.get(v, 0.0) for v in available_views], dtype=float)
            area_target = float(np.dot(weights, area_targets))
            area_cur = abs(_polygon_area(interpolated))
            if area_cur > 1e-8 and area_target > 1e-8:
                scale = math.sqrt(area_target / area_cur)
                c = np.mean(interpolated, axis=0, keepdims=True)
                interpolated = (interpolated - c) * scale + c
            return interpolated
        return self._interpolate_nearest_edge(stroke, target_view)

    def create_group(self, group_id: str, members: List[str], parent_group: Optional[str] = None):
        """Create a group of strokes or sub-groups."""
        if group_id in self.groups:
            return
        group = StrokeGroup(group_id)
        for m in members:
            if m in self.strokes:
                stroke = self.strokes[m]
                stroke.group_id = group_id
                group.strokes.append(stroke)
            elif m in self.groups:
                sub_group = self.groups[m]
                sub_group.group_id = parent_group or group_id
                group.strokes.extend(sub_group.strokes)
        self.groups[group_id] = group
        anchors = [s.anchor_3d for s in group.strokes if s.anchor_3d is not None]
        if anchors:
            group.anchor_3d = np.mean(anchors, axis=0)
        else:
            group.anchor_3d = self.default_group_anchor

    def render_novel_view(self, camera: Camera) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Render a novel view by interpolating shapes."""
        rendered = {}
        for stroke_id, stroke in self.strokes.items():
            shape = self.interpolate_shape(stroke_id, camera)
            if shape is not None:
                rendered[stroke_id] = shape
        target_view = (camera.yaw, camera.pitch)
        closest = self._find_closest_view(*target_view)
        order = self.view_orders.get(closest, list(rendered.keys()))
        order = [sid for sid in order if sid in rendered]
        missing = [sid for sid in rendered if sid not in order]
        order += missing
        return rendered, order

def parse_svg_file(svg_file: str, view_prefix: str = "") -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, str]], List[str]]:
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    strokes = {}
    styles = {}
    order = []

    def _parse_float_with_units(value: str, default: float) -> float:
        try:
            numeric = re.sub(r'[^\d.-]', '', value)
            return float(numeric)
        except (ValueError, TypeError):
            return default

    def _parse_transform(transform: str) -> np.ndarray:
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
        if points.size == 0:
            return points
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
        transformed = points_homogeneous @ transform.T
        return transformed[:, :2]

    def _get_style(elem):
        style = {'fill': 'none', 'stroke': '#000000', 'stroke-width': '1'}
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

    def _sample_path(d: str, samples_per_seg: int = 100) -> List[List[float]]:
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
        parent_map = {c: p for p in tree.iter() for c in p}
        for elem in root.iter():
            if not elem.tag.endswith(('path', 'ellipse', 'circle')):
                continue
            transform = _parse_transform(elem.get('transform'))
            current = parent_map.get(elem)
            while current is not None:
                transform = _parse_transform(current.get('transform')) @ transform
                if current == root:
                    break
                current = parent_map.get(current)
            transform = transform @ viewbox_matrix
            style = _get_style(elem)
            current = parent_map.get(elem)
            while current is not None:
                ancestor_style = _get_style(current)
                for k, v in ancestor_style.items():
                    if k not in style or style[k] == 'none' or style[k] == 'inherit':
                        style[k] = v
                if current == root:
                    break
                current = parent_map.get(current)
            original_id = elem.get('id')
            stroke_id = f"{view_prefix}{original_id}" if original_id else f"{view_prefix}shape_{len(order)}"
            points = None
            if elem.tag.endswith('path'):
                d = elem.get('d', '')
                pts_list = _sample_path(d)
                if pts_list:
                    points = np.array(pts_list, dtype=float)
                    points = _apply_transform(points, transform)
                    style['original_d'] = d
            elif elem.tag.endswith('ellipse'):
                try:
                    cx = _parse_float_with_units(elem.get('cx', '0'), 0)
                    cy = _parse_float_with_units(elem.get('cy', '0'), 0)
                    rx = _parse_float_with_units(elem.get('rx', '0'), 0)
                    ry = _parse_float_with_units(elem.get('ry', '0'), 0)
                    pts = []
                    for angle in np.linspace(0, 2*np.pi, 64, endpoint=True):
                        x = cx + rx * np.cos(angle)
                        y = cy + ry * np.sin(angle)
                        pts.append([x, y])
                    points = np.array(pts, dtype=float)
                    points = _apply_transform(points, transform)
                except ValueError:
                    continue
            elif elem.tag.endswith('circle'):
                try:
                    cx = _parse_float_with_units(elem.get('cx', '0'), 0)
                    cy = _parse_float_with_units(elem.get('cy', '0'), 0)
                    r = _parse_float_with_units(elem.get('r', '0'), 0)
                    pts = []
                    for angle in np.linspace(0, 2*np.pi, 64, endpoint=True):
                        x = cx + r * np.cos(angle)
                        y = cy + r * np.sin(angle)
                        pts.append([x, y])
                    points = np.array(pts, dtype=float)
                    points = _apply_transform(points, transform)
                except ValueError:
                    continue
            if points is not None and len(points) > 0:
                strokes[stroke_id] = points
                styles[stroke_id] = style
                order.append(stroke_id)
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

    stroke_ids = order if order else list(strokes.keys())
    for sid in stroke_ids:
        if sid in strokes:
            g = ET.SubElement(root, "g", attrib={"id": sid})
            style = styles.get(sid, {})
            path_d = _path_d(strokes[sid], close_path=style.get('fill', 'none') != 'none')
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
                if view == (math.pi, 0) or view[1] == -math.pi/2:
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
    additional_yaws = [math.pi/2, -math.pi/2, math.pi]
    
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
    """Visualize the Parameterized Orientation Space with character renderings."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    yaws = [view[0] for view in model.key_views]
    pitches = [view[1] for view in model.key_views]
    colors = []
    for view in model.key_views:
        if view in [(0, 0), (0, math.pi/2), (-math.pi/2, 0)]:
            colors.append('red')
        elif view in [(math.pi, 0), (math.pi/2, 0), (0, -math.pi/2)]:
            colors.append('green')
        else:
            colors.append('blue')
    scatter = ax.scatter(yaws, pitches, c=colors, s=100, alpha=0.7)
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
    ax.set_xlabel('Yaw (radians)')
    ax.set_ylabel('Pitch (radians)')
    ax.set_title('Parameterized Orientation Space (12 Views)')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Input Views'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Mirrored Views'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Rotated Views')
    ]
    ax.legend(handles=legend_elements, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-math.pi/2, math.pi/2)
    overview_path = os.path.join(output_dir, "orientation_space_overview.png")
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Orientation space overview saved to {overview_path}")
    
    print("Creating individual view visualizations for the 12 main views...")
    main_views = [
        (0, 0), (0, math.pi/2), (-math.pi/2, 0),
        (math.pi, 0), (math.pi/2, 0), (0, -math.pi/2),
        (math.pi/2, math.pi/2), (-math.pi/2, math.pi/2), (math.pi, math.pi/2),
        (math.pi/2, -math.pi/2), (-math.pi/2, -math.pi/2), (math.pi, -math.pi/2)
    ]
    main_views = [view for view in main_views if view in model.key_views]
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    for i, (yaw, pitch) in enumerate(main_views[:12]):
        ax = axes[i]
        view_strokes = {}
        for stroke_id, stroke in model.strokes.items():
            if (yaw, pitch) in stroke.key_views:
                view_strokes[stroke_id] = stroke.key_views[(yaw, pitch)]
        print(f"Rendering view (yaw={yaw:.2f}, pitch={pitch:.2f}) with {len(view_strokes)} strokes")
        base_order = model.view_orders.get((yaw, pitch), [])
        ordered_ids = [sid for sid in base_order if sid in view_strokes]
        missing = [sid for sid in view_strokes if sid not in ordered_ids]
        ordered_ids += sorted(missing)
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
        view_name = f"Yaw: {math.degrees(yaw):.0f}°, Pitch: {math.degrees(pitch):.0f}°"
        if (yaw, pitch) in important_views:
            view_name = f"{important_views[(yaw, pitch)]}\n{view_name}"
        ax.set_title(view_name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(len(main_views[:12]), len(axes)):
        axes[i].set_visible(False)
    grid_path = os.path.join(output_dir, "12_main_views_grid.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"12 main views grid saved to {grid_path}")
    
    for i, (yaw, pitch) in enumerate(main_views[:12]):
        fig, ax = plt.subplots(figsize=(8, 8))
        view_strokes = {}
        for stroke_id, stroke in model.strokes.items():
            if (yaw, pitch) in stroke.key_views:
                view_strokes[stroke_id] = stroke.key_views[(yaw, pitch)]
        base_order = model.view_orders.get((yaw, pitch), [])
        ordered_ids = [sid for sid in base_order if sid in view_strokes]
        missing = [sid for sid in view_strokes if sid not in ordered_ids]
        ordered_ids += sorted(missing)
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
        view_name = f"Yaw: {math.degrees(yaw):.0f}°, Pitch: {math.degrees(pitch):.0f}°"
        if (yaw, pitch) in important_views:
            view_name = f"{important_views[(yaw, pitch)]} - {view_name}"
        ax.set_title(view_name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        view_filename = f"view_{i:02d}_{important_views.get((yaw, pitch), 'unknown').lower().replace(' ', '_')}.png"
        view_path = os.path.join(output_dir, view_filename)
        plt.savefig(view_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"All individual views saved to {output_dir}")
    
    print("\n=== Checking original input views ===")
    original_views_to_check = [(0, 0), (0, math.pi/2), (-math.pi/2, 0)]
    original_view_names = ["Front", "Top", "Right"]
    for (yaw, pitch), name in zip(original_views_to_check, original_view_names):
        if (yaw, pitch) in model.key_views:
            stroke_count = sum(1 for stroke_id, stroke in model.strokes.items() if (yaw, pitch) in stroke.key_views)
            print(f"{name} view (yaw={math.degrees(yaw):.0f}°, pitch={math.degrees(pitch):.0f}°): {stroke_count} strokes")
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
    
    front_strokes, front_styles, front_order = parse_svg_file('data/yellow_head_front.svg')
    top_strokes, top_styles, top_order = parse_svg_file('data/yellow_head_top.svg')
    right_strokes, right_styles, right_order = parse_svg_file('data/yellow_head_right.svg')
    
    print(f"Front strokes: {len(front_strokes)}, Order: {front_order}")
    print(f"Top strokes: {len(top_strokes)}, Order: {top_order}")
    print(f"Right strokes: {len(right_strokes)}, Order: {right_order}")
    
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
    
    model.create_group('eyes', ['rightEye', 'leftEye'])
    model.create_group('earsFace', ['rightEar', 'leftEar', 'face'], 'head')
    model.create_group('head', ['eyes', 'earsFace', 'nose', 'mouth'])
    
    novel_camera = Camera(yaw=math.pi/6, pitch=math.pi/6)  # Adjusted to valid angle
    rendered, novel_order = model.render_novel_view(novel_camera)
    export_svg(rendered, front_styles, model.image_width, model.image_height, "novel_view.svg", novel_order)
    
    print("2.5D model created successfully!")
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()