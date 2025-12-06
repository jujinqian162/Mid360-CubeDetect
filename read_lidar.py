import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import PointCloud2 
import sensor_msgs_py.point_cloud2 as pc2 
import numpy as np
import sys
import os

# 添加 CubeDetect 模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CubeDetect.extract_face import detect_cube_from_points


class LivoxSubscriber(Node): 
    def __init__(self): 
        super().__init__('livox_subscriber') 
        # 订阅 /livox/lidar 话题 
        self.subscription = self.create_subscription( 
            PointCloud2, 
            '/livox/lidar', 
            self.listener_callback, 
            10) 
        self.subscription  # prevent unused variable warning 
        
        self.frame_count = 0
        self.detect_interval = 10  # 每 10 帧检测一次
        
        print("正在等待雷达数据...") 
        print("正方体检测已启用 (每10帧检测一次)")

    def listener_callback(self, msg): 
        # 收到数据时的回调 
        n_points = msg.width * msg.height 
        self.frame_count += 1
        
        # 读取点云数据
        points_list = []
        gen = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True) 
        for p in gen:
            points_list.append([p[0], p[1], p[2], p[3]])
        
        points = np.array(points_list, dtype=np.float32)
        
        print(f"--> 帧 {self.frame_count}: 收到 {len(points)} 个有效点")
        
        # 每隔一定帧数进行正方体检测
        if self.frame_count % self.detect_interval == 0 and len(points) > 100:
            print("    [检测中] 正在进行正方体检测...")
            result = detect_cube_from_points(points)
            
            if result['cube_found']:
                print(f"    [检测结果] 发现正方体!")
                print(f"      - 正方体中心: ({result['cube_center'][0]:.3f}, {result['cube_center'][1]:.3f}, {result['cube_center'][2]:.3f})")
                print(f"      - 检测到 {result['num_faces']} 个面")
                for i, fc in enumerate(result['face_centers']):
                    print(f"      - 面{i+1}中心: ({fc[0]:.3f}, {fc[1]:.3f}, {fc[2]:.3f})")
            else:
                print("    [检测结果] 未检测到正方体")
        
        print("-" * 50) 

def main(args=None): 
    rclpy.init(args=args) 
    node = LivoxSubscriber() 
     
    try: 
        rclpy.spin(node) 
    except KeyboardInterrupt: 
        pass 
     
    node.destroy_node() 
    rclpy.shutdown() 

if __name__ == '__main__': 
    main()
