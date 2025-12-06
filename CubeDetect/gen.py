import numpy as np 
import open3d as o3d 
import random 
 
def generate_rough_ground(center_x, center_y, width=8.0, resolution=0.1): 
    """生成高度不平整的地面""" 
    x_range = np.arange(center_x - width/2, center_x + width/2, resolution) 
    y_range = np.arange(center_y - width/2, center_y + width/2, resolution) 
    grid_x, grid_y = np.meshgrid(x_range, y_range) 
    z = np.zeros_like(grid_x) 
     
    # 地形混合 
    freq1 = random.uniform(0.1, 0.3) 
    amp1 = random.uniform(0.1, 0.3) 
    z += np.sin(grid_x * freq1) * np.cos(grid_y * freq1) * amp1 
     
    freq2 = random.uniform(0.8, 1.5) 
    amp2 = random.uniform(0.02, 0.08) 
    z += np.sin(grid_x * freq2 + random.uniform(0, 10)) * np.cos(grid_y * freq2) * amp2 
 
    num_bumps = random.randint(3, 8)  
    for _ in range(num_bumps): 
        bx = random.uniform(x_range[0], x_range[-1]) 
        by = random.uniform(y_range[0], y_range[-1]) 
        if bx**2 + by**2 < 1.0: continue 
        h = random.uniform(0.1, 0.4) 
        w = random.uniform(0.3, 0.8) 
        dist_sq = (grid_x - bx)**2 + (grid_y - by)**2 
        z += h * np.exp(-dist_sq / (2 * w**2)) 
 
    noise = np.random.normal(0, 0.025, z.shape)  
    z += noise 
 
    # 中心平坦化 
    dist_to_center = np.sqrt((grid_x - center_x)**2 + (grid_y - center_y)**2) 
    flat_radius = 0.6 
    transition_width = 0.6 
    mask = np.clip((dist_to_center - flat_radius) / transition_width, 0, 1) 
    center_h_approx = np.mean(z[dist_to_center < flat_radius + 0.1]) 
    z = z * mask + center_h_approx * (1 - mask) 
 
    ground_points = np.stack((grid_x.flatten(), grid_y.flatten(), z.flatten()), axis=1) 
     
    # 碎石 
    debris_points = [] 
    num_debris = random.randint(5, 15) 
    for _ in range(num_debris): 
        dx = random.uniform(x_range[0], x_range[-1]) 
        dy = random.uniform(y_range[0], y_range[-1]) 
        if dx**2 + dy**2 < 0.8: continue 
        ix = int((dx - x_range[0]) / resolution) 
        iy = int((dy - y_range[0]) / resolution) 
        if 0 <= ix < z.shape[1] and 0 <= iy < z.shape[0]: 
            base_z = z[iy, ix] 
            rock_size = random.uniform(0.05, 0.15) 
            for _ in range(random.randint(5, 15)): 
                rx = dx + random.gauss(0, rock_size/2) 
                ry = dy + random.gauss(0, rock_size/2) 
                rz = base_z + random.uniform(0, rock_size) 
                debris_points.append([rx, ry, rz]) 
     
    if debris_points: 
        ground_points = np.vstack((ground_points, np.array(debris_points))) 
 
    return ground_points, center_h_approx 
 
def generate_random_walls(num_walls, area_width, ground_center_h): 
    """生成随机墙壁""" 
    walls_points = [] 
    for _ in range(num_walls): 
        length = random.uniform(2.0, 4.0) 
        height = random.uniform(1.0, 2.0) 
         
        # 重试机制确保墙壁在远处 
        max_attempts = 50 
        placed = False 
        for _ in range(max_attempts): 
            cx = random.uniform(-area_width/2 * 0.9, area_width/2 * 0.9) 
            cy = random.uniform(-area_width/2 * 0.9, area_width/2 * 0.9) 
            # 距离中心至少 2.5 米 (场地变小了，限制稍微放宽一点) 
            if cx**2 + cy**2 > 2.5**2: 
                placed = True 
                break 
         
        if not placed: 
            continue 
         
        yaw = random.uniform(0, 2 * np.pi) 
        points_per_unit = 500 
        num_pts = int(length * height * points_per_unit) 
         
        lx = np.random.uniform(-length/2, length/2, num_pts) 
        lz = np.random.uniform(0, height, num_pts) 
        ly = np.zeros(num_pts) 
 
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]) 
        pts = np.stack((lx, ly, lz), axis=1) @ Rz.T 
        pts[:, 0] += cx 
        pts[:, 1] += cy 
        pts[:, 2] += ground_center_h - 0.1  
        walls_points.append(pts) 
     
    if not walls_points: return np.empty((0, 3)) 
    all_wall_points = np.vstack(walls_points) 
    noise = np.random.normal(0, 0.02, all_wall_points.shape) 
    all_wall_points += noise 
    return all_wall_points 
 
def generate_noisy_cube(z_offset, num_points=2048): 
    """生成立方体""" 
    base_size = random.uniform(0.6, 1.2) 
    size_x = base_size * random.uniform(0.8, 1.2) 
    size_y = base_size * random.uniform(0.8, 1.2) 
    size_z = base_size * random.uniform(0.8, 1.2) 
     
    points_per_face = num_points // 6 
    points = [] 
     
    def get_grid(sx, sy): 
        ax_x = np.linspace(-sx/2, sx/2, int(np.sqrt(points_per_face))) 
        ax_y = np.linspace(-sy/2, sy/2, int(np.sqrt(points_per_face))) 
        gx, gy = np.meshgrid(ax_x, ax_y) 
        return gx.flatten(), gy.flatten() 
 
    fx, fy = get_grid(size_x, size_y) 
    points.append(np.stack((fx, fy, np.ones_like(fx) * size_z/2), axis=1))  
    points.append(np.stack((fx, fy, -np.ones_like(fx) * size_z/2), axis=1))  
     
    fx, fz = get_grid(size_x, size_z) 
    points.append(np.stack((fx, np.ones_like(fx) * size_y/2, fz), axis=1)) 
    points.append(np.stack((fx, -np.ones_like(fx) * size_y/2, fz), axis=1)) 
     
    fy, fz = get_grid(size_y, size_z) 
    points.append(np.stack((np.ones_like(fy) * size_x/2, fy, fz), axis=1)) 
    points.append(np.stack((-np.ones_like(fy) * size_x/2, fy, fz), axis=1)) 
 
    cube_points = np.vstack(points) 
    noise = np.random.normal(0, 0.015, cube_points.shape) 
    cube_points += noise 
     
    dropout_rate = random.uniform(0.0, 0.3) 
    if dropout_rate > 0: 
        mask = np.random.rand(len(cube_points)) > dropout_rate 
        cube_points = cube_points[mask] 
 
    yaw = random.uniform(0, 2 * np.pi) 
    pitch = random.uniform(-0.15, 0.15) 
    roll = random.uniform(-0.15, 0.15) 
 
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]) 
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]) 
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]) 
     
    cube_points = cube_points @ (Rz @ Ry @ Rx).T 
    cube_points[:, 2] += (size_z / 2.0 + z_offset - 0.03) 
     
    return cube_points, max(size_x, size_y, size_z) 
 
def get_camera_position(target_center, dist_range=(3.0, 6.0)): 
    dist = random.uniform(*dist_range) 
    azimuth = random.uniform(0, 2 * np.pi) 
    elevation_deg = random.uniform(0, 30) 
    elevation = np.radians(elevation_deg) 
     
    cam_z = dist * np.sin(elevation) + target_center[2] 
    radius_xy = dist * np.cos(elevation) 
    cam_x = radius_xy * np.cos(azimuth) + target_center[0] 
    cam_y = radius_xy * np.sin(azimuth) + target_center[1] 
     
    return [cam_x, cam_y, cam_z] 
 
def transform_to_sensor_frame(pcd, cam_pos, target_center): 
    """ 
    将点云从世界坐标系转换到传感器(相机)坐标系 
    """ 
    cam_pos = np.array(cam_pos) 
    target = np.array(target_center) 
     
    # Forward (Z) 
    forward = target - cam_pos 
    forward = forward / np.linalg.norm(forward) 
     
    world_up = np.array([0, 0, 1]) 
     
    # Right (X) 
    right = np.cross(forward, world_up) 
    if np.linalg.norm(right) < 1e-6: 
        right = np.array([1, 0, 0]) 
    right = right / np.linalg.norm(right) 
     
    # Up (Y - adjusted to be intuitive Up for visualization relative to camera) 
    cam_up = np.cross(right, forward) 
    cam_up = cam_up / np.linalg.norm(cam_up) 
     
    # 3x3 Rotation 
    R = np.array([right, cam_up, forward])  
     
    points = np.asarray(pcd.points) 
    points = points - cam_pos # Translate 
    points = points @ R.T     # Rotate 
     
    pcd.points = o3d.utility.Vector3dVector(points) 
    return pcd 
 
def generate_scene(ground_width=8.0, enable_walls=True, max_walls=2): 
    """ 
    生成场景的主函数 
    :param ground_width: 地面宽度，默认为 8.0 
    :param enable_walls: 是否生成墙壁 
    :param max_walls: 生成墙壁的最大数量 
    """ 
    MAX_SCENE_RETRIES = 3 
    MAX_CAM_RETRIES = 10 
    MIN_VISIBLE_POINTS = 50  
     
    for scene_idx in range(MAX_SCENE_RETRIES): 
        ground_pts, z_offset = generate_rough_ground(0, 0, width=ground_width, resolution=0.1) 
         
        # --- 墙壁配置逻辑 --- 
        if enable_walls: 
            num_walls = random.randint(1, max_walls) # 随机生成 1 到 max_walls 面墙 
            wall_pts = generate_random_walls(num_walls, ground_width, z_offset) 
        else: 
            wall_pts = [] # 不生成墙壁 
        # ------------------ 
         
        cube_pts, cube_max_size = generate_noisy_cube(z_offset, num_points=2500) 
        num_cube_points = len(cube_pts) 
         
        if len(wall_pts) > 0: 
            all_points = np.vstack((cube_pts, ground_pts, wall_pts)) 
        else: 
            all_points = np.vstack((cube_pts, ground_pts)) 
 
        pcd = o3d.geometry.PointCloud() 
        pcd.points = o3d.utility.Vector3dVector(all_points) 
        target_center = [0, 0, z_offset + cube_max_size/2] 
         
        for cam_idx in range(MAX_CAM_RETRIES): 
            cam_pos = get_camera_position(target_center) 
             
            _, pt_map = pcd.hidden_point_removal(cam_pos, radius=1000) 
             
            visible_cube_count = sum(1 for idx in pt_map if idx < num_cube_points) 
             
            if visible_cube_count > MIN_VISIBLE_POINTS: 
                print(f"成功生成! 场景重试:{scene_idx}, 视角重试:{cam_idx}, 可见立方体点数:{visible_cube_count}") 
                 
                pcd_visible = pcd.select_by_index(pt_map) 
                pcd_visible = pcd_visible.voxel_down_sample(voxel_size=0.04) 
                 
                print("正在将点云转换到传感器(相机)坐标系...") 
                pcd_visible = transform_to_sensor_frame(pcd_visible, cam_pos, target_center) 
                 
                pts = np.asarray(pcd_visible.points) 
                if len(pts) > 0: 
                    z_min, z_max = np.min(pts[:, 2]), np.max(pts[:, 2]) 
                    z_norm = (pts[:, 2] - z_min) / (z_max - z_min + 1e-6) 
                else: 
                    z_norm = np.zeros_like(pts[:, 2]) 
 
                colors = np.ones_like(pts) * 0.5  
                colors += np.random.uniform(-0.2, 0.2, size=(len(pts), 1))  
                colors += z_norm[:, np.newaxis] * 0.2 
                colors = np.clip(colors, 0, 1) 
                 
                pcd_visible.colors = o3d.utility.Vector3dVector(colors) 
                 
                return pcd_visible, [0, 0, 0] 
            else: 
                pass 
         
        print("当前场景难以找到观测点，重新生成场景布局...") 
 
    raise RuntimeError("无法生成有效数据：立方体总是被严重遮挡。") 
 
if __name__ == "__main__": 
    output_filename = "scanned_cube.ply" 
     
    # --- 配置选项 --- 
    CONFIG = { 
        "ground_width": 8.0,   # 场地大小，默认改回 8.0 
        "enable_walls": False,  # 墙壁开关 
        "max_walls": 2         # 最大墙壁数量 (之前是5，现在减小) 
    } 
    # ---------------- 
     
    try: 
        pcd, cam_pos = generate_scene(**CONFIG) 
         
        with open("camera_pos.txt", "w") as f: 
            f.write(f"{cam_pos[0]},{cam_pos[1]},{cam_pos[2]}") 
             
        o3d.io.write_point_cloud(output_filename, pcd, write_ascii=True) 
        print(f"生成完毕: {output_filename}") 
        print("注意：现在点云已转换为传感器坐标系，相机位于 (0,0,0)。") 
    except Exception as e: 
        print(f"生成失败: {e}") 
