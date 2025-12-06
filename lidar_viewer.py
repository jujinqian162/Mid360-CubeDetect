import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import PointCloud2 
import sensor_msgs_py.point_cloud2 as pc2 
import open3d as o3d 
import numpy as np 
import threading 
import time 
 
class RealTimeVisualizer(Node): 
    def __init__(self): 
        super().__init__('lidar_visualizer') 
         
        # 1. 初始化 Open3D 窗口 
        self.vis = o3d.visualization.Visualizer() 
        self.vis.create_window(window_name="Livox Mid-360 Stream", width=960, height=720) 
         
        # 2. 创建一个空的点云对象 
        self.pcd = o3d.geometry.PointCloud() 
        # 先加一个假点，防止 Open3D 报错 
        self.pcd.points = o3d.utility.Vector3dVector(np.array([[0,0,0]], dtype=np.float64)) 
        self.vis.add_geometry(self.pcd) 
 
        # 设置渲染选项（比如点的大小，背景色） 
        opt = self.vis.get_render_option() 
        if opt is not None:
            opt.background_color = np.asarray([0, 0, 0]) # 黑色背景 
            opt.point_size = 2.0 
        else:
            print("warn: 无法获取渲染选项，使用默认背景")
 
        # 3. 订阅雷达数据 
        self.subscription = self.create_subscription( 
            PointCloud2, 
            '/livox/lidar', 
            self.lidar_callback, 
            10 
        ) 
         
        # 用于线程间传递数据的缓冲区 
        self.latest_points = None 
        self.lock = threading.Lock() 
 
    def lidar_callback(self, msg): 
        # 将 ROS 消息转换为 numpy 数组 (x, y, z) 
        # 注意：这步比较耗时，尽量优化 
        gen = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True) 
        raw_data = list(gen) 
         
        if not raw_data: 
            return 
 
        # 转换为 numpy 矩阵 
        np_data = np.array(raw_data, dtype=np.float32) 
         
        # 提取坐标 (前3列) 和 强度 (第4列) 
        points = np_data[:, :3] 
        intensity = np_data[:, 3] 
 
        with self.lock: 
            self.latest_points = (points, intensity) 
 
    def update_vis(self): 
        # 这是主渲染循环，相当于 cv2.imshow 的刷新逻辑 
        with self.lock: 
            if self.latest_points is not None: 
                points, intensity = self.latest_points 
                 
                # 更新点坐标 
                self.pcd.points = o3d.utility.Vector3dVector(points) 
 
                # 把强度映射为颜色 (简单的灰度映射) 
                # 归一化强度到 0-1 之间 
                if intensity.max() > 0: 
                    intensity_norm = intensity / 255.0  # Livox 强度通常在 0-255 
                else: 
                    intensity_norm = intensity 
                 
                # 创建颜色数组 (N, 3)，这里用一种伪彩色 (橙色到黄色) 
                colors = np.zeros((points.shape[0], 3)) 
                colors[:, 0] = intensity_norm        # R 
                colors[:, 1] = intensity_norm * 0.5  # G 
                colors[:, 2] = intensity_norm * 0.2  # B 
                 
                self.pcd.colors = o3d.utility.Vector3dVector(colors) 
 
                # 通知 Open3D 数据变了 
                self.vis.update_geometry(self.pcd) 
                self.latest_points = None 
 
        # Open3D 标准渲染流程 
        self.vis.poll_events() 
        self.vis.update_renderer() 
 
def main(args=None): 
    rclpy.init(args=args) 
    node = RealTimeVisualizer() 
 
    # 使用单独的线程跑 ROS 接收，防止界面卡死 
    # 这样主线程可以专心负责渲染 
    ros_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True) 
    ros_thread.start() 
 
    print("开始渲染循环... 按 'Q' 退出窗口") 
    try: 
        while True: 
            node.update_vis() 
            # 控制帧率，避免 CPU 100% 
            time.sleep(0.01)  
    except KeyboardInterrupt: 
        pass 
    finally: 
        node.vis.destroy_window() 
        rclpy.shutdown() 
 
if __name__ == '__main__': 
    main()
