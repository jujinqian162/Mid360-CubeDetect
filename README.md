# Mid360-Lidar 立方体检测

本仓库聚焦于两部分：
1. `CubeDetect/` 中的算法核⼼，用于在 Livox Mid360 点云中自动定位立方体及其可见面心。
2. 仓库根目录中的 ROS2 订阅/工具脚本，用于采集传感器数据并把外层消息输送给算法。

---

## 1. CubeDetect 算法模块

### 1.1 模块概述
`CubeDetect` 提供点云预处理、地面分割、DBSCAN 聚类、立方体尺寸筛选与平面提取等功能，可直接输出最近面的中心坐标，适合机器人抓取、姿态估计等场景。

### 1.2 目录与依赖
- 关键脚本：`extract_face.py`（检测与可视化流程）、`gen*.py`（生成模拟点云）、`visu.py` 等。
- 依赖：Python 3.10+、[Open3D](http://www.open3d.org/)、NumPy；推荐 `pip install -e CubeDetect` 以便调试。

### 1.3 快速开始
```bash
cd CubeDetect
python extract_face.py        # 默认读取 scanned_cube.ply
```
程序会自动去除地面、聚类物体、筛选尺寸合适的立方体，并展示可视化结果。

代码方式调用：
```python
from CubeDetect.extract_face import detect_cube_from_points

result = detect_cube_from_points(points,
                                 min_cube_size=0.3,
                                 max_cube_size=0.7)
if result["cube_found"]:
    print("最近面心:", result["face_centers"][0])
```

### 1.4 尺寸筛选与参数
- `min_cube_size` / `max_cube_size`：以聚类的轴对齐包围盒 (AABB) 最大边长为依据，默认 0.3 m ~ 0.7 m。
- 其他可调：`ground_distance_threshold`（地面分割阈值）、`cluster_eps` 与 `cluster_min_points`（聚类密度）、`face_distance_threshold`（平面分割精度）。

### 1.5 可视化提示
- 灰色：原始场景点云
- 红色：检测到的立方体可见面
- 绿色球：对应面心
- 坐标轴：雷达坐标系

---

## 2. ROS2 订阅与外围脚本

### 2.1 文件概览
- `read_lidar.py` / `read_coustom_msg.py`：示例 ROS2 节点，订阅 `sensor_msgs/PointCloud2` 或自定义消息，转为 `numpy` 点云。
- `lidar_viewer.py` / `lidar_text.py`：轻量化的点云可视化与调试工具，便于在离线数据上复现算法表现。

### 2.2 使用流程
1. **启动 ROS2 订阅节点**：根据实际雷达话题修改 `read_lidar.py` 中的话题名（如 `/livox/lidar`），运行 `python read_lidar.py` 获取实时点云。
2. **数据桥接**：在回调中将 `PointCloud2` 转为 `numpy.ndarray`，再调用 `detect_cube_from_points()` 将结果（面心、立方体中心等）发布到新的话题或服务。
3. **可视化/日志**：利用 `lidar_viewer.py` 或 `lidar_text.py` 检查订阅到的帧是否完整，并可把同一帧存为 `.ply` 以便后续离线分析。

### 2.3 推荐集成方式
- 在 ROS2 节点中提前创建 `CubeDetect` 的参数配置（如尺寸范围、聚类阈值），确保线上与离线调试保持一致。
- 对于不同场景可建立多套参数（室内、室外、远距离模式），并通过 ROS 参数服务器或 YAML 动态加载。
- 如需多帧融合，可在外层对接缓存/滤波节点，再把融合后的点云传入 `CubeDetect`，提高鲁棒性。

如需进一步扩展（例如添加 ROS2 action、输出姿态估计等），可在第二部分脚本基础上添加新的订阅/发布逻辑，与 `CubeDetect` 保持解耦。
