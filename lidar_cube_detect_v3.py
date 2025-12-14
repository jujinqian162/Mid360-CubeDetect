"""
LiDAR Cube Detection V3 - 使用改进的V3算法进行立方体检测

运行方式:
    python lidar_cube_detect_v3.py
"""

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

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from CubeDetect.extract_face_v3 import detect_cube_v3, CubeDetectorConfigV3
except ImportError as e:
    print(f"Error: Could not import V3 detector. {e}")
    sys.exit(1)


class LidarCubeVisualizerV3(Node):
    def __init__(self):
        super().__init__('lidar_cube_visualizer_v3')
        
        # 订阅雷达话题
        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.listener_callback,
            10
        )
        
        self.latest_points = None
        self.lock = threading.Lock()
        
        print("="*50)
        print("  LiDAR Cube Detection V3 已启动")
        print("="*50)
        print("正在等待雷达数据...")
        print()
        print("可视化说明:")
        print("  灰色点 = 原始点云")
        print("  红色球 = 检测到的立方体中心")
        print("  绿色框 = OBB包围盒")
        print("  黄色球 = 次佳候选")
        print()

    def listener_callback(self, msg):
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        points_list = []
        for p in gen:
            points_list.append([float(p[0]), float(p[1]), float(p[2])])
        
        if not points_list:
            return

        points_np = np.array(points_list, dtype=np.float32)
        
        with self.lock:
            self.latest_points = points_np


def create_wireframe_box(center, extent, color=[0, 1, 0]):
    """创建线框包围盒"""
    half = np.array(extent) / 2.0
    
    # 8个顶点
    corners = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ]) * half + center
    
    # 12条边
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 侧边
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    
    return line_set


def main(args=None):
    rclpy.init(args=args)
    node = LidarCubeVisualizerV3()
    
    # ROS线程
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    
    # Open3D可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Cube Detection V3", width=1280, height=960)
    
    # 主点云
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    # 存储动态几何体
    geometry_store = []
    
    # 渲染选项
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0.05, 0.05, 0.1])  # 深蓝黑色背景
    
    # 配置V3参数
    config = CubeDetectorConfigV3()
    config.max_range = 4.0
    config.min_range = 0.3
    config.min_cube_size = 0.15
    config.max_cube_size = 0.8
    config.target_cube_size = 0.5
    config.cluster_eps = 0.08
    config.cluster_min_points = 15
    config.ground_height_threshold = 0.10
    
    print(f"V3 配置:")
    print(f"  范围: {config.min_range}m - {config.max_range}m")
    print(f"  目标尺寸: {config.target_cube_size}m")
    print(f"  聚类eps: {config.cluster_eps}")
    print()
    
    first_update = True
    frame_count = 0
    last_print_time = time.time()
    
    try:
        while True:
            if not vis.poll_events():
                break
            
            with node.lock:
                points = node.latest_points
                node.latest_points = None
            
            if points is not None:
                frame_count += 1
                
                # 更新点云
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.paint_uniform_color([0.6, 0.6, 0.6])
                vis.update_geometry(pcd)
                
                # V3检测
                try:
                    result = detect_cube_v3(points, config=config)
                    
                    # 移除旧几何体
                    for geo in geometry_store:
                        vis.remove_geometry(geo, reset_bounding_box=False)
                    geometry_store = []
                    
                    if result['cube_found']:
                        candidates = result['candidates']
                        
                        # 过滤掉置信度太低的候选
                        good_candidates = [c for c in candidates if c['confidence'] in ['HIGH', 'MEDIUM']]
                        
                        # 定期打印检测结果
                        current_time = time.time()
                        if current_time - last_print_time > 2.0:  # 每2秒打印一次
                            print(f"\n[Frame {frame_count}] 检测到 {len(candidates)} 个候选, {len(good_candidates)} 个有效:")
                            for idx, cand in enumerate(candidates[:5]):
                                center = cand['cube_center']
                                score = cand['score']
                                conf = cand['confidence']
                                pts = cand['num_points']
                                ext = cand.get('extent', [0, 0, 0])
                                marker = "★" if conf in ['HIGH', 'MEDIUM'] else " "
                                print(f"  {marker}#{idx+1}: 中心=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
                                      f"分数={score:.1f}, 置信={conf}, 点数={pts}, 尺寸={np.mean(ext):.2f}m")
                            last_print_time = current_time
                        
                        # 可视化：优先显示高置信度的候选
                        display_candidates = good_candidates[:2] if good_candidates else candidates[:1]
                        
                        for idx, cand in enumerate(display_candidates):
                            center = np.array(cand['cube_center'])
                            extent = cand.get('extent', [0.5, 0.5, 0.5])
                            conf = cand['confidence']
                            
                            if conf == 'HIGH':
                                # 高置信度: 红色中心，亮绿色框
                                sphere_color = [1.0, 0.0, 0.0]
                                box_color = [0.0, 1.0, 0.0]
                                sphere_radius = 0.06
                            elif conf == 'MEDIUM':
                                # 中等置信度: 橙色中心，黄绿色框
                                sphere_color = [1.0, 0.5, 0.0]
                                box_color = [0.5, 1.0, 0.0]
                                sphere_radius = 0.05
                            else:
                                # 低置信度: 黄色中心，蓝色框
                                sphere_color = [1.0, 1.0, 0.0]
                                box_color = [0.3, 0.3, 1.0]
                                sphere_radius = 0.04
                            
                            # 中心球
                            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
                            sphere.translate(center)
                            sphere.paint_uniform_color(sphere_color)
                            vis.add_geometry(sphere, reset_bounding_box=False)
                            geometry_store.append(sphere)
                            
                            # 包围盒
                            box = create_wireframe_box(center, extent, box_color)
                            vis.add_geometry(box, reset_bounding_box=False)
                            geometry_store.append(box)
                            
                            # 面中心 - 只为第一个（最佳）候选显示
                            if idx == 0:
                                for fc in cand.get('face_centers', [])[:2]:
                                    fc_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
                                    fc_sphere.translate(np.array(fc))
                                    fc_sphere.paint_uniform_color([0.0, 1.0, 1.0])  # 青色
                                    vis.add_geometry(fc_sphere, reset_bounding_box=False)
                                    geometry_store.append(fc_sphere)
                    
                except Exception as e:
                    print(f"检测错误: {e}")
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
