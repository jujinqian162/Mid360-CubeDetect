import rclpy
from rclpy.node import Node
# 关键变化 1: 导入 Livox 自定义消息类型
from livox_ros_driver2.msg import CustomMsg

class LivoxCustomReader(Node):
    def __init__(self):
        super().__init__('livox_custom_reader')
        
        # 关键变化 2: 订阅 CustomMsg 类型
        self.subscription = self.create_subscription(
            CustomMsg,
            '/livox/lidar',
            self.listener_callback,
            10
        )
        self.subscription  # 防止被垃圾回收
        print("正在等待 Livox CustomMsg 格式数据...")

    def listener_callback(self, msg):
        # msg 的结构非常简单直接
        # msg.header: 标准头
        # msg.point_num: 这一帧有多少个点
        # msg.points: 一个列表，包含所有点
        
        points_count = msg.point_num
        print(f"--> 收到一帧数据！包含 {points_count} 个点")

        # 打印前 5 个点的信息
        # 自定义格式的点包含: x, y, z, reflectivity (反射率), tag, line, offset_time
        count = 0
        for p in msg.points:
            # 注意：这里直接用 .x, .y, .z 访问，非常方便
            print(f"    点{count}: x={p.x:.3f}, y={p.y:.3f}, z={p.z:.3f}, 强度={p.reflectivity}")
            
            count += 1
            if count >= 5:
                break
        print("-" * 30)

def main(args=None):
    rclpy.init(args=args)
    node = LivoxCustomReader()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
