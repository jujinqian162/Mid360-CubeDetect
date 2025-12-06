import numpy as np 
import open3d as o3d 
import random 
 
def get_box_points(size_x, size_y, size_z, resolution=0.05): 
    """ 
    生成一个标准长方体的表面点云 (中心在原点) 
    """ 
    # 计算各面需要的点数 
    points = [] 
     
    # 辅助函数：生成平面的网格点 
    def make_plane(u_range, v_range, w_val, axis_idx): 
        # axis_idx: 0=x固定, 1=y固定, 2=z固定 
        u_grid, v_grid = np.meshgrid( 
            np.arange(u_range[0], u_range[1], resolution), 
            np.arange(v_range[0], v_range[1], resolution) 
        ) 
        u = u_grid.flatten() 
        v = v_grid.flatten() 
        w = np.full_like(u, w_val) 
         
        if axis_idx == 0:   return np.stack((w, u, v), axis=1) # x fixed, y=u, z=v 
        elif axis_idx == 1: return np.stack((u, w, v), axis=1) # y fixed, x=u, z=v 
        else:               return np.stack((u, v, w), axis=1) # z fixed, x=u, y=v 
 
    # 生成 5 个面 (底面往往看不见，为了效率可以不生成，或者为了完整性生成) 
    # Top (+Z) 
    points.append(make_plane((-size_x/2, size_x/2), (-size_y/2, size_y/2), size_z/2, 2)) 
    # Front (+Y) 
    points.append(make_plane((-size_x/2, size_x/2), (-size_z/2, size_z/2), size_y/2, 1)) 
    # Back (-Y) 
    points.append(make_plane((-size_x/2, size_x/2), (-size_z/2, size_z/2), -size_y/2, 1)) 
    # Right (+X) 
    points.append(make_plane((-size_y/2, size_y/2), (-size_z/2, size_z/2), size_x/2, 0)) 
    # Left (-X) 
    points.append(make_plane((-size_y/2, size_y/2), (-size_z/2, size_z/2), -size_x/2, 0)) 
     
    # 稍微加一点点随机扰动，消除完美网格感 
    all_pts = np.vstack(points) 
    all_pts += np.random.normal(0, 0.005, all_pts.shape) 
     
    return all_pts 
 
def generate_block_ground(area_size=10.0, block_base_size=1.2): 
    """ 
    生成由高低错落的长方体组成的地面 
    返回: 
    1. 地面点云数组 
    2. valid_spots: 列表，包含每个方块顶面的中心坐标 [x, y, z]，用于放置物体 
    """ 
    ground_points = [] 
    valid_spots = [] 
     
    # 计算网格数量 
    steps = int(area_size / block_base_size) 
    start = -area_size / 2 + block_base_size / 2 
     
    print(f"正在生成错落方块地形 ({steps}x{steps})...") 
     
    for i in range(steps): 
        for j in range(steps): 
            cx = start + i * block_base_size 
            cy = start + j * block_base_size 
             
            # 随机高度: 基础高度 + 随机偏移 
            # 修改：降低高度差，由 0.6 改为 0.2，使地形更平缓 
            height_variation = block_base_size * 0.35 
            top_z = random.uniform(-height_variation, height_variation) 
             
            # 柱子总高度 (为了不露馅，让它向下延伸很多) 
            total_h = top_z - (-3.0)  
            center_z = (-3.0 + top_z) / 2.0 
             
            # 生成柱子点云 (分辨率稍微低一点以减少总点数) 
            pts = get_box_points(block_base_size * 0.95, block_base_size * 0.95, total_h, resolution=0.1) 
             
            # 平移到指定位置 
            pts[:, 0] += cx 
            pts[:, 1] += cy 
            pts[:, 2] += center_z 
             
            ground_points.append(pts) 
             
            # 记录顶面中心，供放置正方体使用 
            valid_spots.append({'x': cx, 'y': cy, 'z': top_z}) 
     
    all_ground_pts = np.vstack(ground_points) 
     
    # 修改：增加地面噪声，让表面更粗糙 
    all_ground_pts += np.random.normal(0, 0.02, all_ground_pts.shape) 
             
    return all_ground_pts, valid_spots 
 
def generate_target_cube(cx, cy, bottom_z, num_points=2000): 
    """ 
    在指定位置生成一个目标正方体 
    """ 
    # 随机尺寸 
    base_size = random.uniform(0.5, 0.9) 
    # 允许轻微长方 
    sx = base_size * random.uniform(0.9, 1.1) 
    sy = base_size * random.uniform(0.9, 1.1) 
    sz = base_size * random.uniform(0.9, 1.1) 
     
    # 生成点 (复用之前的逻辑，或者直接用 get_box_points 并加密) 
    pts = get_box_points(sx, sy, sz, resolution=0.03) # 目标物体稍微密一点 
     
    # 1. 加噪声 
    pts += np.random.normal(0, 0.01, pts.shape) 
     
    # 2. 随机旋转 (Yaw, Pitch, Roll) 
    yaw = random.uniform(0, 2 * np.pi) 
    pitch = random.uniform(-0.1, 0.1) # 轻微倾斜 
    roll = random.uniform(-0.1, 0.1) 
     
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]) 
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]) 
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]) 
     
    pts = pts @ (Rz @ Ry @ Rx).T 
     
    # 3. 平移放置 
    # 原始中心在 0,0,0。要让底部 (z = -sz/2) 贴合 bottom_z 
    # 所以 z_center 应该是 bottom_z + sz/2 
    pts[:, 0] += cx 
    pts[:, 1] += cy 
    pts[:, 2] += (bottom_z + sz/2.0) 
     
    return pts, np.array([cx, cy, bottom_z + sz/2.0]) 
 
def get_camera_position(target_center, dist_range=(4.0, 7.0)): 
    dist = random.uniform(*dist_range) 
    azimuth = random.uniform(0, 2 * np.pi) 
    elevation_deg = random.uniform(10, 45) # 稍微高一点的视角方便看清错落的方块 
    elevation = np.radians(elevation_deg) 
     
    cam_z = dist * np.sin(elevation) + target_center[2] 
    radius_xy = dist * np.cos(elevation) 
    cam_x = radius_xy * np.cos(azimuth) + target_center[0] 
    cam_y = radius_xy * np.sin(azimuth) + target_center[1] 
     
    return [cam_x, cam_y, cam_z] 
 
def transform_to_sensor_frame(pcd, cam_pos, target_center): 
    """世界坐标 -> 传感器坐标""" 
    cam_pos = np.array(cam_pos) 
    target = np.array(target_center) 
     
    forward = target - cam_pos 
    forward = forward / np.linalg.norm(forward) 
     
    # 防止看天或看地导致的奇异 
    right = np.cross(forward, np.array([0, 0, 1])) 
    if np.linalg.norm(right) < 0.01: 
        right = np.array([1, 0, 0]) 
    right = right / np.linalg.norm(right) 
     
    cam_up = np.cross(right, forward) 
    cam_up = cam_up / np.linalg.norm(cam_up) 
     
    R = np.array([right, cam_up, forward]) 
     
    points = np.asarray(pcd.points) 
    points = points - cam_pos 
    points = points @ R.T 
     
    pcd.points = o3d.utility.Vector3dVector(points) 
    return pcd 
 
def generate_scene(): 
    MAX_RETRIES = 5 
     
    for attempt in range(MAX_RETRIES): 
        # 1. 生成方块地面 
        ground_pts, valid_spots = generate_block_ground(area_size=10.0, block_base_size=1.2) 
         
        # 修改：为地面生成颜色 (灰色) 
        ground_colors = np.full(ground_pts.shape, 0.5) 
        # 给地面颜色加点杂色 
        ground_colors += np.random.uniform(-0.1, 0.1, ground_colors.shape) 
         
        # 2. 随机选择 1-3 个位置放置目标 
        num_targets = random.randint(1, 3) 
        print(f"计划生成 {num_targets} 个目标正方体...") 
         
        # 从 valid_spots 中随机选，不重复 
        if len(valid_spots) < num_targets: 
            continue 
         
        selected_spots = random.sample(valid_spots, num_targets) 
         
        cube_points_list = [] 
        target_centers = [] 
         
        for spot in selected_spots: 
            c_pts, c_center = generate_target_cube(spot['x'], spot['y'], spot['z']) 
            cube_points_list.append(c_pts) 
            target_centers.append(c_center) 
             
        # 合并所有点云 
        all_cube_pts = np.vstack(cube_points_list) 
         
        # 修改：为正方体生成颜色 (红色，高亮显示) 
        cube_colors = np.tile(np.array([1.0, 0.0, 0.0]), (all_cube_pts.shape[0], 1)) 
         
        # 合并点和颜色 
        all_points = np.vstack((ground_pts, all_cube_pts)) 
        all_colors = np.vstack((ground_colors, cube_colors)) 
        all_colors = np.clip(all_colors, 0, 1) 
         
        # 创建 PointCloud 
        pcd = o3d.geometry.PointCloud() 
        pcd.points = o3d.utility.Vector3dVector(all_points) 
        pcd.colors = o3d.utility.Vector3dVector(all_colors) # 设置颜色 
         
        # 3. 确定相机目标点 (所有目标的几何中心) 
        avg_target_center = np.mean(target_centers, axis=0) 
         
        # 4. 获取相机位置 & HPR 
        cam_pos = get_camera_position(avg_target_center) 
         
        _, pt_map = pcd.hidden_point_removal(cam_pos, radius=1000) 
        pcd_visible = pcd.select_by_index(pt_map) 
         
        # 简单检查可见性 (点数是否急剧减少) 
        if len(pcd_visible.points) < len(all_points) * 0.1: 
            print("可见点太少，重试...") 
            continue 
 
        # 降采样 (这会自动混合颜色) 
        pcd_visible = pcd_visible.voxel_down_sample(voxel_size=0.04) 
         
        # 5. 坐标转换 
        pcd_visible = transform_to_sensor_frame(pcd_visible, cam_pos, avg_target_center) 
         
        # 6. 上色 (之前是重新上色，现在已经有颜色了，所以这步省略，直接返回带有红色的点云) 
             
        return pcd_visible, [0,0,0] # 相机已转为原点 
 
    raise RuntimeError("生成失败，请重试") 
 
if __name__ == "__main__": 
    output_filename = "scanned_cube.ply" 
    try: 
        pcd, cam_pos = generate_scene() 
        o3d.io.write_point_cloud(output_filename, pcd, write_ascii=True) 
        # 为了 extract_face.py 能正常工作，我们还是写一个假的 camera_pos 
        with open("camera_pos.txt", "w") as f: 
            f.write("0,0,0")  
        print(f"生成完毕: {output_filename}") 
        print("场景说明: 错落方块地形(低落差) + 随机1~3个正方体(红色)") 
    except Exception as e: 
        print(f"Error: {e}") 
