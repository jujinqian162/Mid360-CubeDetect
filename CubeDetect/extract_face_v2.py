import open3d as o3d
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

@dataclass
class CubeDetectorConfig:
    """
    Configuration for CubeDetectorV2.
    """
    # 1. Range Filtering
    min_range: float = 0.2
    max_range: float = 4.0 
    
    # 2. Ground Removal
    ground_distance_threshold: float = 0.08
    ground_ransac_n: int = 3
    ground_iterations: int = 500
    
    # 3. Clustering (DBSCAN)
    cluster_eps: float = 0.45      
    cluster_min_points: int = 20   # Reduced to catch sparse cubes
    
    # 4. Cube Sizing
    # User target is ~0.5m. 
    # Relax min to 0.20 to catch "partial" cubes (e.g. top half only)
    min_cube_size: float = 0.20    
    max_cube_size: float = 1.0     
    
    # 5. Face Segmentation
    face_distance_threshold: float = 0.06 # Tightened slightly to prefer flat things
    min_face_points: int = 10       
    max_faces_to_check: int = 5    
    
    # 6. Geometric Logic
    angle_tolerance_deg: float = 25.0 
    
    def get_ortho_threshold(self) -> float:
        return math.sin(math.radians(self.angle_tolerance_deg))


@dataclass
class CubeCandidate:
    """
    Represents a detected cube candidate.
    """
    center: np.ndarray             
    face_centers: List[np.ndarray] 
    face_normals: List[np.ndarray] 
    num_faces: int                 
    num_points: int                
    distance: float                
    score: float                   
    confidence_info: str           
    
    def to_dict(self):
        return {
            'cube_center': self.center.tolist(),
            'face_centers': [fc.tolist() for fc in self.face_centers],
            'num_faces': self.num_faces,
            'num_points': self.num_points,
            'distance': self.distance,
            'score': self.score,
            'confidence_info': self.confidence_info
        }


class CubeDetectorV2:
    def __init__(self, config: CubeDetectorConfig = None):
        if config is None:
            self.config = CubeDetectorConfig()
        else:
            self.config = config

    def detect(self, points: np.ndarray) -> List[CubeCandidate]:
        if points is None or len(points) == 0:
            return []
            
        # Ensure Nx3
        if points.shape[1] > 3:
            points = points[:, :3]

        # --- Step 1: Range Filtering ---
        dists = np.linalg.norm(points, axis=1)
        valid_mask = (dists >= self.config.min_range) & (dists <= self.config.max_range)
        filtered_points = points[valid_mask]
        
        if len(filtered_points) < self.config.cluster_min_points:
            return []

        # Convert to Open3D cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # --- Step 2: Ground Removal ---
        try:
            _, inliers = pcd.segment_plane(
                distance_threshold=self.config.ground_distance_threshold,
                ransac_n=self.config.ground_ransac_n,
                num_iterations=self.config.ground_iterations
            )
            non_ground_pcd = pcd.select_by_index(inliers, invert=True)
        except Exception:
            non_ground_pcd = pcd

        if len(non_ground_pcd.points) < self.config.cluster_min_points:
            return []

        # --- Step 3: Clustering ---
        labels = np.array(non_ground_pcd.cluster_dbscan(
            eps=self.config.cluster_eps,
            min_points=self.config.cluster_min_points,
            print_progress=False
        ))

        if labels.size == 0:
            return []
        
        max_label = labels.max()
        if max_label < 0:
            return []

        candidates = []
        ortho_thresh = self.config.get_ortho_threshold()

        # Iterate over clusters
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            num_cluster_points = len(cluster_indices)
            
            if num_cluster_points < self.config.cluster_min_points:
                continue
            
            cluster_pcd = non_ground_pcd.select_by_index(cluster_indices)
            cluster_center = cluster_pcd.get_center()
            dist_to_sensor = np.linalg.norm(cluster_center)

            # --- Step 4: Size & Shape Checks ---
            aabb = cluster_pcd.get_axis_aligned_bounding_box()
            extent = aabb.get_extent()
            sorted_extent = sorted(extent) # [min, mid, max]
            min_dim, mid_dim, max_dim = sorted_extent[0], sorted_extent[1], sorted_extent[2]
            
            # Debug Print (Uncomment if needed)
            # print(f"ID:{i} Pts:{num_cluster_points} Dim:{max_dim:.2f}x{mid_dim:.2f}x{min_dim:.2f}")

            # 4a. Absolute Size Limit
            if max_dim > self.config.max_cube_size or max_dim < self.config.min_cube_size:
                continue
            
            # 4b. Aspect Ratio - A cube shouldn't be a long pole
            # Relaxed slightly to 3.5 to allow for partial views
            if max_dim > 3.5 * mid_dim:
                continue

            # --- Step 5: Plane Extraction ---
            detected_planes = [] 
            temp_pcd = cluster_pcd
            
            for _ in range(self.config.max_faces_to_check):
                if len(temp_pcd.points) < self.config.min_face_points:
                    break
                
                try:
                    plane_model, inliers = temp_pcd.segment_plane(
                        distance_threshold=self.config.face_distance_threshold,
                        ransac_n=3,
                        num_iterations=50
                    )
                    
                    if len(inliers) < self.config.min_face_points:
                        break
                        
                    normal = np.array(plane_model[:3])
                    norm_len = np.linalg.norm(normal)
                    if norm_len == 0: 
                        break
                    normal = normal / norm_len
                    
                    face_subset = temp_pcd.select_by_index(inliers)
                    face_center = face_subset.get_center()
                    
                    detected_planes.append({
                        'normal': normal,
                        'center': face_center,
                        'count': len(inliers)
                    })
                    
                    temp_pcd = temp_pcd.select_by_index(inliers, invert=True)
                except Exception:
                    break
            
            # --- Step 6: Classification & Scoring ---
            
            orthogonal_pairs_count = 0
            valid_faces_indices = set()
            
            if len(detected_planes) >= 2:
                for idx1 in range(len(detected_planes)):
                    for idx2 in range(idx1 + 1, len(detected_planes)):
                        n1 = detected_planes[idx1]['normal']
                        n2 = detected_planes[idx2]['normal']
                        if abs(np.dot(n1, n2)) < ortho_thresh:
                            orthogonal_pairs_count += 1
                            valid_faces_indices.add(idx1)
                            valid_faces_indices.add(idx2)

            final_faces = []
            confidence_info = ""
            base_score = 0.0
            
            # Classification Logic
            is_valid_candidate = False

            if orthogonal_pairs_count > 0:
                # --- Tier 1: Orthogonal (Best) ---
                confidence_info = "ORTHO_MATCH"
                base_score = 1000.0 + (orthogonal_pairs_count * 100)
                final_faces = [detected_planes[i] for i in valid_faces_indices]
                is_valid_candidate = True
                
            elif len(detected_planes) >= 1:
                # --- Tier 2: Single Face (Good) ---
                # Key requirement: If we only see one face, the object effectively IS that face.
                # So the points should mostly belong to that plane.
                total_plane_points = sum(p['count'] for p in detected_planes)
                plane_ratio = total_plane_points / num_cluster_points
                
                # If > 50% of points are on planes, it's a flat-faced object (good).
                # If < 30%, it's a messy blob with a small accidental plane (noise).
                if plane_ratio > 0.4:
                    confidence_info = "SINGLE_FACE"
                    base_score = 500.0
                    final_faces = detected_planes[:1]
                    is_valid_candidate = True
                else:
                    # Too messy to be a clean cube face
                     confidence_info = "MESSY_FACE_REJECTED"
            
            else:
                # --- Tier 3: Shape Only (Fallback) ---
                # STRICT check: To accept a shape-only match, it must be dense and cubic.
                
                # 1. 3D Structure: Must not be flat
                if min_dim < 0.1: 
                     confidence_info = "TOO_FLAT"
                # 2. Cubic Aspect: max/mid ratio close to 1
                elif max_dim > 2.5 * mid_dim:
                     confidence_info = "NOT_SQUARE"
                # 3. Density/Count: Must have enough points to be sure
                elif num_cluster_points < 50:
                     confidence_info = "TOO_SPARSE_FOR_SHAPE"
                else:
                    confidence_info = "SHAPE_ONLY"
                    base_score = 100.0
                    final_faces = []
                    is_valid_candidate = True

            if not is_valid_candidate:
                continue
            
            # Limit faces for visualization
            final_faces.sort(key=lambda x: x['count'], reverse=True)
            final_faces = final_faces[:3]
            
            score = base_score
            score += num_cluster_points * 0.1
            score -= dist_to_sensor * 10.0
            
            cand = CubeCandidate(
                center=cluster_center,
                face_centers=[f['center'] for f in final_faces],
                face_normals=[f['normal'] for f in final_faces],
                num_faces=len(final_faces),
                num_points=num_cluster_points,
                distance=dist_to_sensor,
                score=score,
                confidence_info=confidence_info
            )
            candidates.append(cand)

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

def detect_cube_v2_simple(points, config=None):
    if config is None:
        config = CubeDetectorConfig()
    detector = CubeDetectorV2(config)
    candidates = detector.detect(points)
    
    return {
        'cube_found': len(candidates) > 0,
        'candidates': [c.to_dict() for c in candidates]
    }
