"""
使用 python-pcl 库进行正方体检测
需要安装: pip install python-pcl 或 conda install -c conda-forge python-pcl
"""
import numpy as np

try:
    import pcl
    PCL_AVAILABLE = True
except ImportError:
    PCL_AVAILABLE = False
    print("警告: python-pcl 未安装，请运行: pip install python-pcl")


def detect_cube_from_points_pcl(points,
                                 ground_distance_threshold=0.08,
                                 cluster_tolerance=0.20,
                                 cluster_min_size=30,
                                 cluster_max_size=50000,
                                 face_distance_threshold=0.04,
                                 min_face_points=50):
    """
    使用 PCL 库从点云数据中检测正方体并提取面中心
    
    Args:
        points: numpy array (N, 3) 或 (N, 4)，点云数据
        ground_distance_threshold: 地面分割阈值 (RANSAC)
        cluster_tolerance: 欧式聚类容差
        cluster_min_size: 聚类最小点数
        cluster_max_size: 聚类最大点数
        face_distance_threshold: 面分割阈值
        min_face_points: 最小面点数
    
    Returns:
        dict: {
            'cube_found': bool,
            'face_centers': list of [x, y, z],
            'num_faces': int,
            'cube_center': [x, y, z] or None
        }
    """
    result = {
        'cube_found': False,
        'face_centers': [],
        'num_faces': 0,
        'cube_center': None
    }
    
    if not PCL_AVAILABLE:
        print("错误: python-pcl 未安装")
        return result
    
    if len(points) < 100:
        return result
    
    # 只取 xyz
    if points.shape[1] > 3:
        points = points[:, :3]
    
    points = points.astype(np.float32)
    
    # 创建 PCL 点云
    cloud = pcl.PointCloud()
    cloud.from_array(points)
    
    # 1. 使用 RANSAC 去除地面 (分割最大平面)
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(ground_distance_threshold)
    
    indices, coefficients = seg.segment()
    
    if len(indices) > 0:
        # 提取非地面点
        objects_cloud = cloud.extract(indices, negative=True)
    else:
        objects_cloud = cloud
    
    if objects_cloud.size < cluster_min_size:
        return result
    
    # 2. 欧式聚类分离物体
    tree = objects_cloud.make_kdtree()
    ec = objects_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(cluster_tolerance)
    ec.set_MinClusterSize(cluster_min_size)
    ec.set_MaxClusterSize(cluster_max_size)
    ec.set_SearchMethod(tree)
    
    cluster_indices = ec.Extract()
    
    if len(cluster_indices) == 0:
        return result
    
    # 3. 找最可能是正方体的聚类 (离中心轴最近)
    cube_cloud = None
    min_dist = float('inf')
    
    for indices in cluster_indices:
        if len(indices) < 100:
            continue
        
        cluster_cloud = objects_cloud.extract(indices)
        cluster_points = np.asarray(cluster_cloud)
        center = np.mean(cluster_points, axis=0)
        dist = np.sqrt(center[0]**2 + center[1]**2)
        
        if dist < min_dist:
            min_dist = dist
            cube_cloud = cluster_cloud
            cube_center = center
    
    if cube_cloud is None:
        return result
    
    result['cube_center'] = cube_center.tolist()
    
    # 4. 提取正方体的面
    remaining_cloud = cube_cloud
    face_centers = []
    
    for _ in range(3):  # 最多提取 3 个面
        if remaining_cloud.size < min_face_points:
            break
        
        # RANSAC 平面分割
        seg = remaining_cloud.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(face_distance_threshold)
        
        indices, coefficients = seg.segment()
        
        if len(indices) < min_face_points:
            break
        
        # 提取面点云
        face_cloud = remaining_cloud.extract(indices)
        face_points = np.asarray(face_cloud)
        face_center = np.mean(face_points, axis=0)
        face_centers.append(face_center.tolist())
        
        # 去除已提取的面，继续找下一个面
        remaining_cloud = remaining_cloud.extract(indices, negative=True)
    
    if face_centers:
        result['cube_found'] = True
        result['face_centers'] = face_centers
        result['num_faces'] = len(face_centers)
    
    return result


def extract_nearest_face_pcl(filename="scanned_cube.ply"):
    """
    从 PLY 文件读取点云并检测正方体
    """
    if not PCL_AVAILABLE:
        print("错误: python-pcl 未安装")
        return
    
    print(f"正在读取 {filename} ...")
    
    try:
        cloud = pcl.load(filename)
    except Exception as e:
        print(f"无法读取 {filename}: {e}")
        return
    
    if cloud.size == 0:
        print("点云为空！")
        return
    
    points = np.asarray(cloud)
    print(f"读取到 {len(points)} 个点")
    
    result = detect_cube_from_points_pcl(points)
    
    if result['cube_found']:
        print(f"检测到正方体!")
        print(f"  - 中心: {result['cube_center']}")
        print(f"  - 检测到 {result['num_faces']} 个面")
        for i, fc in enumerate(result['face_centers']):
            print(f"  - 面{i+1}中心: ({fc[0]:.3f}, {fc[1]:.3f}, {fc[2]:.3f})")
    else:
        print("未检测到正方体")
    
    return result


if __name__ == "__main__":
    extract_nearest_face_pcl()
