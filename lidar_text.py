import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class LidarTextPrinter(Node):
    def __init__(self):
        super().__init__('lidar_text_printer')
        # 订阅 /livox/lidar 话题
        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.listener_callback,
            10
        )
        self.subscription  # 防止被垃圾回收
        print("节点已启动，正在等待雷达数据...")

    def listener_callback(self, msg):
        # 1. 获取这一帧的总点数
        n_points = msg.width * msg.height
        
        # 2. 读取点云数据 (生成器)
        # field_names 必须和雷达发布的一致，通常 Livox 发布的是 x, y, z, intensity
        gen = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        
        print(f"\n[新的一帧] 时间戳: {msg.header.stamp.sec}.{msg.header.stamp.nanosec} | 总点数: {n_points}")
        print("-" * 50)
        print(f"{'X (m)':<10} | {'Y (m)':<10} | {'Z (m)':<10} | {'强度':<10}")
        print("-" * 50)

        # 3. 只打印前 10 个点，避免刷屏卡死终端
        count = 0
        for p in gen:
            # p[0]=x, p[1]=y, p[2]=z, p[3]=intensity
            print(f"{p[0]:<10.3f} | {p[1]:<10.3f} | {p[2]:<10.3f} | {p[3]:<10.0f}")
            
            count += 1
            if count >= 10:
                print(f"... (还有 {n_points - 10} 个点未显示)")
                break

def main(args=None):
    rclpy.init(args=args)
    node = LidarTextPrinter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n程序停止")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
