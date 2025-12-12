import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import threading
import time
import sys
import os

# Add the current directory to sys.path to ensure we can import CubeDetect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from CubeDetect.extract_face_v2 import detect_cube_v2_simple, CubeDetectorConfig
except ImportError as e:
    print(f"Error: Could not import detect_cube_v2_simple from CubeDetect.extract_face_v2. {e}")
    sys.exit(1)

class LidarCubeVisualizer(Node):
    def __init__(self):
        super().__init__('lidar_cube_visualizer')
        # 订阅 /livox/lidar 话题
        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.listener_callback,
            10
        )
        self.latest_points = None
        self.lock = threading.Lock()
        print("节点已启动 (V2 Logic)，正在等待雷达数据...")
        print("可视化: 灰色 = 原始点云")
        print("        红色球 = 最佳立方体中心")
        print("        绿色球 = 最佳立方体的面中心")
        print("        黄色球 = 次佳候选")

    def listener_callback(self, msg):
        # 读取点云数据
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        points_list = []
        for p in gen:
            points_list.append([float(p[0]), float(p[1]), float(p[2])])
        
        if not points_list:
            return

        points_np = np.array(points_list, dtype=np.float32)
        
        with self.lock:
            self.latest_points = points_np

def main(args=None):
    rclpy.init(args=args)
    node = LidarCubeVisualizer()
    
    # 在单独的线程中运行 ROS spin，以便主线程用于 Open3D 可视化
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    
    # 初始化 Open3D Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Cube Detection V2", width=1024, height=768)
    
    # 1. 主点云 (灰色)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    # 2. 几何体管理
    geometry_store = [] # 用于存储当前显示的球体等几何体

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1]) # 深灰背景
    
    # 配置 V2 参数
    config = CubeDetectorConfig()
    config.max_range = 3.0       # 3米以外忽略
    config.min_range = 0.2
    config.min_cube_size = 0.15
    config.max_cube_size = 1.0
    config.cluster_eps = 0.35    # 稍微调大一点适应稀疏点云
    config.ground_distance_threshold = 0.05
    
    print(f"--- Configuration ---")
    print(f"Max Range: {config.max_range}m")
    print(f"Cube Size: {config.min_cube_size}-{config.max_cube_size}m")
    
    first_update = True
    
    try:
        while True:
            # 保持窗口响应
            if not vis.poll_events():
                break
                
            with node.lock:
                points = node.latest_points
                node.latest_points = None # 读取后清空
            
            if points is not None:
                # 更新主点云
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.paint_uniform_color([0.6, 0.6, 0.6]) # 灰色
                vis.update_geometry(pcd)
                
                # 进行正方体检测
                try:
                    result = detect_cube_v2_simple(points, config=config)
                    
                    # --- 更新可视化 ---
                    # 1. 移除旧的几何体
                    for geo in geometry_store:
                        vis.remove_geometry(geo, reset_bounding_box=False)
                    geometry_store = []
                    
                    if result['cube_found'] and result['candidates']:
                        candidates = result['candidates']
                        
                        # 可视化所有候选者 (Top 3)
                        for idx, cand in enumerate(candidates[:3]):
                            cube_center = cand['cube_center']
                            face_centers = cand['face_centers']
                            
                            # 颜色方案
                            if idx == 0:
                                # 最佳: 红色中心，绿色面心
                                center_color = [1.0, 0.0, 0.0]
                                face_color = [0.0, 1.0, 0.0]
                                center_radius = 0.08
                                face_radius = 0.05
                            else:
                                # 其他: 黄色中心，蓝色面心
                                center_color = [1.0, 1.0, 0.0]
                                face_color = [0.0, 0.0, 1.0]
                                center_radius = 0.06
                                face_radius = 0.04

                            # 绘制立方体中心
                            sphere_c = o3d.geometry.TriangleMesh.create_sphere(radius=center_radius)
                            sphere_c.translate(cube_center)
                            sphere_c.paint_uniform_color(center_color)
                            vis.add_geometry(sphere_c, reset_bounding_box=False)
                            geometry_store.append(sphere_c)

                            # 绘制面中心
                            for fc in face_centers:
                                sphere_f = o3d.geometry.TriangleMesh.create_sphere(radius=face_radius)
                                sphere_f.translate(fc)
                                sphere_f.paint_uniform_color(face_color)
                                vis.add_geometry(sphere_f, reset_bounding_box=False)
                                geometry_store.append(sphere_f)
                                
                                # 可选：画线连接中心和面心，增强立体感
                                line_points = [cube_center, fc]
                                line_set = o3d.geometry.LineSet()
                                line_set.points = o3d.utility.Vector3dVector(line_points)
                                line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
                                line_set.paint_uniform_color(face_color)
                                vis.add_geometry(line_set, reset_bounding_box=False)
                                geometry_store.append(line_set)
                                
                except Exception as e:
                    print(f"Detection Error: {e}")
                    import traceback
                    traceback.print_exc()

                if first_update:
                    vis.reset_view_point(True)
                    first_update = False
            
            vis.update_renderer()
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n程序停止")
    finally:
        vis.destroy_window()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
