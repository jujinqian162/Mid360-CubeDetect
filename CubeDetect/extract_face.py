import open3d as o3d 
import numpy as np 
import math

def detect_cube_from_points(points,  
                            ground_distance_threshold=0.08, 
                            cluster_eps=0.40,           
                            cluster_min_points=30,      
                            face_distance_threshold=0.05, 
                            min_face_points=10,         
                            min_cube_size=0.15,         
                            max_cube_size=1.0,
                            angle_tolerance_deg=25.0):  
    """ 
    从点云数据中检测正方体并提取面中心 
    改进版 V4：返回所有候选者列表，按优先级排序
    """ 
    # 初始化返回结构
    result = { 
        'cube_found': False, 
        'candidates': [] # List of dicts
    }
     
    if len(points) < cluster_min_points: 
        return result 
     
    # 只取 xyz 
    if points.shape[1] > 3: 
        points = points[:, :3] 
     
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(points) 
     
    # 1. 去除地面 
    try: 
        plane_model, inliers = pcd.segment_plane( 
            distance_threshold=ground_distance_threshold, 
            ransac_n=3, 
            num_iterations=500 
        ) 
        objects_cloud = pcd.select_by_index(inliers, invert=True) 
    except: 
        objects_cloud = pcd 
     
    if len(objects_cloud.points) < cluster_min_points: 
        return result 
     
    # 2. 欧式聚类 
    labels = np.array(objects_cloud.cluster_dbscan( 
        eps=cluster_eps,  
        min_points=cluster_min_points,  
        print_progress=False 
    )) 
     
    if labels.size == 0 or labels.max() < 0: 
        return result 
     
    # 3. 遍历所有聚类，收集候选者
    ortho_threshold = math.sin(math.radians(angle_tolerance_deg)) 
    candidates_list = []

    for i in range(labels.max() + 1): 
        cluster_indices = np.where(labels == i)[0] 
        if len(cluster_indices) < cluster_min_points: 
            continue 
         
        temp_cloud = objects_cloud.select_by_index(cluster_indices) 
        
        # --- A. 尺寸初筛 ---
        bbox = temp_cloud.get_axis_aligned_bounding_box()
        bbox_extent = bbox.get_extent()
        max_extent = max(bbox_extent)
        
        if max_extent < min_cube_size or max_extent > max_cube_size:
            continue
        
        # --- B. 提取平面 ---
        current_faces_normals = []
        current_faces_centers = []
        remaining_points = temp_cloud
        
        for _ in range(3):
            if len(remaining_points.points) < min_face_points:
                break
            
            try:
                plane, inliers = remaining_points.segment_plane(
                    distance_threshold=face_distance_threshold,
                    ransac_n=3,
                    num_iterations=100
                )
                
                if len(inliers) < min_face_points:
                    break
                    
                normal = np.array(plane[:3])
                norm_len = np.linalg.norm(normal)
                if norm_len > 0:
                    normal = normal / norm_len
                    
                face_pcd = remaining_points.select_by_index(inliers)
                current_faces_normals.append(normal)
                current_faces_centers.append(list(face_pcd.get_center()))
                
                remaining_points = remaining_points.select_by_index(inliers, invert=True)
            except Exception:
                break
        
        num_faces = len(current_faces_normals)
        if num_faces < 2:
            continue
            
        # --- C. 计算垂直配对 ---
        orthogonal_pairs = 0
        for idx1 in range(num_faces):
            for idx2 in range(idx1 + 1, num_faces):
                n1 = current_faces_normals[idx1]
                n2 = current_faces_normals[idx2]
                dot_val = abs(np.dot(n1, n2))
                if dot_val < ortho_threshold:
                    orthogonal_pairs += 1
        
        # --- D. 筛选与评分 ---
        # 只要有一对垂直面，就视为有效候选
        has_ortho = (orthogonal_pairs > 0)
        
        if not has_ortho:
            continue

        center = temp_cloud.get_center()
        dist = np.sqrt(center[0]**2 + center[1]**2)
        num_points = len(cluster_indices)
        
        # 评分: (是否合格, 点数量, -距离)
        score = (has_ortho, num_points, -dist)
        
        candidate_data = {
            'cube_center': list(center),
            'face_centers': current_faces_centers,
            'num_faces': num_faces,
            'orthogonal_pairs': orthogonal_pairs,
            'num_points': num_points,
            'distance': dist,
            'score': score
        }
        candidates_list.append(candidate_data)
            
    # 按分数从高到低排序
    # Python tuple 比较机制：先比 score[0](has_ortho), 再比 score[1](num_points), 再比 score[2](-dist)
    candidates_list.sort(key=lambda x: x['score'], reverse=True)
    
    if candidates_list:
        result['cube_found'] = True
        result['candidates'] = candidates_list
        
    return result 
 
 
def extract_nearest_face(filename="scanned_cube.ply", min_cube_size=0.3, max_cube_size=0.7):
    # 保留此函数用于兼容旧的测试脚本，但其实际逻辑应尽可能复用 detect_cube_from_points
    # 为了简化，这里仅提供一个简单的 wrapper 或保留原逻辑(已省略，重点在于 detect_cube_from_points)
    pass