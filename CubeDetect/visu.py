import open3d as o3d 
import os 
import sys 
 
def view_point_cloud(filename): 
    """读取并显示点云""" 
     
    # 检查文件是否存在 
    if not os.path.exists(filename): 
        print(f"错误: 找不到文件 '{filename}'") 
        print("请先运行 gen.py 生成点云文件。") 
        return 
 
    print(f"正在读取 {filename} ...") 
    pcd = o3d.io.read_point_cloud(filename) 
 
    if pcd.is_empty(): 
        print("错误: 点云文件为空或读取失败。") 
        return 
 
    print("正在打开可视化窗口...") 
    print("提示:") 
    print("  - 鼠标左键: 旋转") 
    print("  - Ctrl + 鼠标左键: 缩放") 
    print("  - Shift + 鼠标左键: 平移") 
     
    # 创建坐标轴网格作为参考 (红色=X, 绿色=Y, 蓝色=Z) 
    # 原点处的坐标轴 
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0]) 
 
    # 运行可视化 
    # draw_geometries 接受一个几何体列表 
    o3d.visualization.draw_geometries( 
        [pcd, mesh_frame],  
        window_name="Open3D - 立方体点云查看器", 
        width=800, 
        height=600, 
        left=50, 
        top=50 
    ) 
 
if __name__ == "__main__": 
    target_file = "scanned_cube.ply" 
     
    # 允许用户通过命令行参数指定文件 
    if len(sys.argv) > 1: 
        target_file = sys.argv[1] 
 
    view_point_cloud(target_file) 
