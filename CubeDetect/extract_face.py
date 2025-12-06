import open3d as o3d 
import numpy as np 
 
 
def detect_cube_from_points(points,  
                            ground_distance_threshold=0.08, 
                            cluster_eps=0.20, 
                            cluster_min_points=30, 
                            face_distance_threshold=0.04, 
                            min_face_points=50): 
    """ 
    从点云数据中检测正方体并提取面中心 
     
    Args: 
        points: numpy array (N, 3) 或 (N, 4)，点云数据 
        ground_distance_threshold: 地面分割阈值 
        cluster_eps: DBSCAN 聚类半径 
        cluster_min_points: DBSCAN 最小点数 
        face_distance_threshold: 面分割阈值 
        min_face_points: 最小面点数 
     
    Returns: 
        dict: { 
            'cube_found': bool, 
            'face_centers': list of [x, y, z], 
            'num_faces': int, 
            'cube_center': [x, y, z] or None, 
            'cube_cloud': PointCloud or None 
        } 
    """ 
    result = { 
        'cube_found': False, 
        'face_centers': [], 
        'num_faces': 0, 
        'cube_center': None, 
        'cube_cloud': None 
    } 
     
    if len(points) < 100: 
        return result 
     
    # 只取 xyz 
    if points.shape[1] > 3: 
        points = points[:, :3] 
     
    # 转换为 Open3D 点云 
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(points) 
     
    # 1. 去除地面 
    try: 
        plane_model, inliers = pcd.segment_plane( 
            distance_threshold=ground_distance_threshold, 
            ransac_n=3, 
            num_iterations=1000 
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
     
    # 3. 找最可能是正方体的聚类 (离中心轴最近) 
    cube_pcd = None 
    min_dist = float('inf') 
     
    for i in range(labels.max() + 1): 
        cluster_indices = np.where(labels == i)[0] 
        if len(cluster_indices) < 100: 
            continue 
         
        temp_cloud = objects_cloud.select_by_index(cluster_indices) 
        center = temp_cloud.get_center() 
        dist = np.sqrt(center[0]**2 + center[1]**2) 
         
        if dist < min_dist: 
            min_dist = dist 
            cube_pcd = temp_cloud 
     
    if cube_pcd is None: 
        return result 
     
    result['cube_center'] = list(cube_pcd.get_center()) 
    result['cube_cloud'] = cube_pcd 
     
    # 4. 提取正方体的面 
    remaining = cube_pcd 
    face_centers = [] 
     
    for _ in range(3):  # 最多提取 3 个面 
        if len(remaining.points) < min_face_points: 
            break 
         
        try: 
            plane_model, inliers = remaining.segment_plane( 
                distance_threshold=face_distance_threshold, 
                ransac_n=3, 
                num_iterations=500 
            ) 
        except: 
            break 
         
        if len(inliers) < min_face_points: 
            break 
         
        face_cloud = remaining.select_by_index(inliers) 
        face_centers.append(list(face_cloud.get_center())) 
        remaining = remaining.select_by_index(inliers, invert=True) 
     
    if face_centers: 
        result['cube_found'] = True 
        result['face_centers'] = face_centers 
        result['num_faces'] = len(face_centers) 
     
    return result 
 
 
def extract_nearest_face(filename="scanned_cube.ply"): 
    print(f"正在读取 {filename} ...") 
     
    # 1. 读取点云 
    try: 
        pcd = o3d.io.read_point_cloud(filename) 
    except: 
        print(f"无法读取 {filename}，请先运行 gen.py 生成数据。") 
        return 
 
    if len(pcd.points) == 0: 
        print("点云为空！") 
        return 
 
    # 预处理：降采样以加快处理速度 (可选) 
    # pcd = pcd.voxel_down_sample(voxel_size=0.02) 
     
    print("正在去除地面...") 
    # 2. 去除地面 (假设地面是场景中最大的平面) 
    # distance_threshold 需要根据 gen.py 中的噪声水平调整，那里噪声约 0.02，所以这里设 0.05 比较稳 
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.08, 
                                           ransac_n=3, 
                                           num_iterations=1000) 
     
    # 地面点 
    # ground_cloud = pcd.select_by_index(inliers) 
    # 剩余物体点 (包含立方体、墙壁、空中的碎石) 
    objects_cloud = pcd.select_by_index(inliers, invert=True) 
     
    if len(objects_cloud.points) < 10: 
        print("警告：去除地面后点数过少，可能地面分割过于激进或正方体太小。") 
        o3d.visualization.draw_geometries([pcd], window_name="原始点云 (分割失败)") 
        return 
 
    print("正在聚类物体...") 
    # 3. 欧式聚类 (DBSCAN) 分离物体 
    # eps: 两个点被视为同一类的最大距离。gen.py 中分辨率约 0.04~0.08，所以 0.15~0.2 比较合适 
    # min_points: 构成一个类所需的最少点数 
    labels = np.array(objects_cloud.cluster_dbscan(eps=0.20, min_points=30, print_progress=False)) 
     
    if labels.size == 0: 
        print("未检测到物体聚类。") 
        return 
 
    max_label = labels.max() 
    print(f"检测到 {max_label + 1} 个潜在物体。") 
 
    # 4. 识别立方体 
    # 策略：摄像机是对着立方体拍的 (Z轴前方)。 
    # 我们找一个点数足够多，且质心距离 Z 轴 (x=0, y=0) 最近的聚类。 
    best_cluster_idx = -1 
    min_dist_to_center_axis = float('inf') 
     
    cube_pcd = None 
     
    for i in range(max_label + 1): 
        # 提取当前聚类的点索引 
        cluster_indices = np.where(labels == i)[0] 
         
        # 过滤掉太小的碎片 (碎石) 
        if len(cluster_indices) < 100: 
            continue 
             
        temp_cloud = objects_cloud.select_by_index(cluster_indices) 
        center = temp_cloud.get_center() 
         
        # 计算质心到 Z 轴的距离 (sqrt(x^2 + y^2)) 
        dist_to_axis = np.sqrt(center[0]**2 + center[1]**2) 
         
        # 优先选离中心轴近的，且距离原点不是特别远或特别近的 
        if dist_to_axis < min_dist_to_center_axis: 
            min_dist_to_center_axis = dist_to_axis 
            best_cluster_idx = i 
            cube_pcd = temp_cloud 
 
    if cube_pcd is None: 
        print("未能识别出立方体 (可能都被过滤掉了)。") 
        # 调试：显示剩余物体 
        objects_cloud.paint_uniform_color([1, 0, 0]) 
        o3d.visualization.draw_geometries([objects_cloud], window_name="所有剩余物体") 
        return 
 
    print("已锁定立方体聚类。正在提取最近面...") 
 
    # 5. 提取立方体的面并找到最近的一个 
    # 立方体可能有1-3个面可见。我们循环分割平面。 
     
    remaining_cube_points = cube_pcd 
    faces = [] 
    face_centers = [] 
     
    # 尝试分割最多 3 个面 
    for i in range(3): 
        if len(remaining_cube_points.points) < 50: 
            break 
             
        # 分割平面 (比地面阈值小一点，因为立方体面比较平，虽然gen加了噪) 
        plane_model, inliers = remaining_cube_points.segment_plane(distance_threshold=0.04, 
                                                                 ransac_n=3, 
                                                                 num_iterations=500) 
         
        if len(inliers) < 50: # 面太小就不算了 
            break 
             
        face_cloud = remaining_cube_points.select_by_index(inliers) 
        faces.append(face_cloud) 
        face_centers.append(face_cloud.get_center()) 
        print(f"提取到第 {i+1} 个面，点数: {len(face_cloud.points)}") 
         
        # 去掉这个面，继续找下一个面 
        remaining_cube_points = remaining_cube_points.select_by_index(inliers, invert=True) 
 
    if not faces: 
        print("未在立方体上检测到平面。") 
        return 
 
    print(f"共提取到 {len(faces)} 个面。") 
 
    # 7. 可视化准备 
    geometries_to_draw = [] 
     
    # a. 背景环境 (变灰，透明度无法直接设置但可以用灰色弱化) 
    pcd.paint_uniform_color([0.8, 0.8, 0.8])  
    geometries_to_draw.append(pcd) 
     
    # b. 提取出的面 (标红) 和面心点 (绿色小球) 
    for i, face in enumerate(faces): 
        face.paint_uniform_color([1, 0, 0]) # Red 
        geometries_to_draw.append(face) 
         
        center = face_centers[i] 
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05) # 稍微调小一点半径 
        sphere.translate(center) 
        sphere.paint_uniform_color([0, 1, 0]) # Green 
        geometries_to_draw.append(sphere) 
 
     
    # d. 添加坐标轴辅助观看 
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0]) 
    geometries_to_draw.append(axes) 
 
    print("可视化中...") 
    print("  - 灰色: 原始场景") 
    print("  - 红色: 提取出的正方体面") 
    print("  - 绿色球: 各面的中心点") 
    print("  - 坐标轴原点: 摄像机位置") 
     
    o3d.visualization.draw_geometries(geometries_to_draw,  
                                    window_name="Open3D - 多面提取", 
                                    width=1000, height=800) 
 
if __name__ == "__main__": 
    extract_nearest_face() 
