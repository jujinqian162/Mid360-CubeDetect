import numpy as np
import open3d as o3d

# 1. 假设这是你的模型生成的数据 (N, 3)
# 例如：生成一个随机的立方体点云
points = np.random.rand(1024, 3).astype(np.float32)

# 2. 转换为 Open3D 的点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 3. (可选) 如果你想在查看器里看到颜色
# 比如把点染成红色 (1, 0, 0)
colors = np.zeros_like(points)
colors[:, 0] = 1  # R通道设为1
pcd.colors = o3d.utility.Vector3dVector(colors)

# 4. 保存文件
# CloudCompare 可以直接打开这个文件
o3d.io.write_point_cloud("my_cube.ply", pcd)
print("保存成功：my_cube.ply")

# 5. (可选) 直接在 Python 中弹窗查看
o3d.visualization.draw_geometries([pcd], window_name="Python Preview")
