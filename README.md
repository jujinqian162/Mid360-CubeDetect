# Mid360-Lidar 立方体检测

本仓库聚焦于两部分：
1. `CubeDetect/` 中的算法核⼼，用于在 Livox Mid360 点云中自动定位立方体及其可见面心。
2. 仓库根目录中的 ROS2 订阅/工具脚本，用于采集传感器数据并把外层消息输送给算法。

---

## 1. CubeDetect 算法模块

### 1.1 模块概述
`CubeDetect` 提供点云预处理、地面分割、DBSCAN 聚类、立方体尺寸筛选与平面提取等功能，可直接输出最近面的中心坐标，适合机器人抓取、姿态估计等场景。

### 1.2 目录与依赖
- 关键脚本：`extract_face.py`（V1检测）、`extract_face_v2.py`（V2检测）、`extract_face_v3.py`（V3检测，推荐）、`gen*.py`（生成模拟点云）、`visu.py` 等。
- 依赖：Python 3.10+、[Open3D](http://www.open3d.org/)、NumPy；推荐 `pip install -e CubeDetect` 以便调试。

### 1.2.1 算法版本说明
- **V1 (`extract_face.py`)**: 基础版本，使用平面分割检测立方体面
- **V2 (`extract_face_v2.py`)**: 改进版，增加正交面验证
- **V3 (`extract_face_v3.py`)**: **推荐使用**，针对稀疏激光雷达优化
  - 简化地面移除（基于高度阈值，避免误删立方体顶面）
  - 自适应多尺度聚类（尝试多个eps值）
  - 基于OBB的形状评估（更好地拟合任意朝向的立方体）
  - 综合评分机制（考虑尺寸、形状、点密度、距离）

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
- `lidar_cube_detect.py`：ROS2 订阅节点（V2算法），直接在回调中调用 `detect_cube_v2_simple()`。
- `lidar_cube_detect_v3.py`：**推荐使用**，ROS2 订阅节点（V3算法），使用改进的检测算法。
- `lidar_viewer.py`：基于 Open3D 的实时可视化节点，用于查看 `/livox/lidar` 点云流并按强度渲染颜色。
- `lidar_text.py`：轻量级文本打印节点，每帧展示前 10 个点，便于快速校验话题数据。
- `read_coustom_msg.py`：订阅 Livox 自定义 `CustomMsg` 的示例，展示如何处理未转换为 PointCloud2 的原始数据。

### 2.2 使用流程
1. **选择订阅节点**：
   - 若需要直接联调检测算法，运行 `python lidar_cube_detect.py`。
   - 若只需查看点云或日志，分别运行 `python lidar_viewer.py` 或 `python lidar_text.py`。
2. **配置话题与频率**：确保所有脚本中的话题名（默认 `/livox/lidar`）与实际系统一致，可根据需要调整检测间隔、可视化刷新频率等参数。
3. **桥接到算法**：订阅节点将 `PointCloud2`/`CustomMsg` 转换为 `numpy.ndarray` 后传入 `detect_cube_from_points()`，你可以在回调里发布新的话题或服务，将面心结果输送给上层控制逻辑。

### 2.3 推荐集成方式
- 在 ROS2 节点中提前创建 `CubeDetect` 的参数配置（如尺寸范围、聚类阈值），确保线上与离线调试保持一致。
- 对于不同场景可建立多套参数（室内、室外、远距离模式），并通过 ROS 参数服务器或 YAML 动态加载。
- 如需多帧融合，可在外层对接缓存/滤波节点，再把融合后的点云传入 `CubeDetect`，提高鲁棒性。

如需进一步扩展（例如添加 ROS2 action、输出姿态估计等），可在第二部分脚本基础上添加新的订阅/发布逻辑，与 `CubeDetect` 保持解耦。
