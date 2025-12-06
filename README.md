# Mid360-Lidar 立方体检测

本仓库用于基于 Livox Mid360 激光雷达点云的机器人立方体识别与面心定位。`CubeDetect` 模块是算法核心，能够在复杂场景中分离目标立方体、提取可见平面，并输出距离最近的面心坐标。

## 仓库结构

- `CubeDetect/`：立方体检测算法及示例脚本（`extract_face.py`、`gen.py` 等）。
- `lidar_viewer.py` / `lidar_text.py`：点云可视化和数据读取辅助工具。
- 其他脚本：与 ROS2 或硬件接口的实验代码。

## 运行依赖

- Python 3.10+
- [Open3D](http://www.open3d.org/)（`pip install open3d`）
- NumPy 等常用科学计算库

> 如果你使用 `pyproject.toml`，可以运行 `pip install -e CubeDetect` 或 `pip install -r requirements.txt`（自行生成）来安装依赖。

## 快速开始

1. **准备点云**：将雷达采集的点云保存为 `.ply` 或转换为 `numpy.ndarray`。
2. **运行示例脚本**：
   ```bash
   cd CubeDetect
   python extract_face.py  # 默认读取 scanned_cube.ply
   ```
   - 程序会自动分割地面、聚类物体、筛选立方体，并显示提取到的面及面心。
3. **在代码中调用**：
   ```python
   from CubeDetect.extract_face import detect_cube_from_points

   result = detect_cube_from_points(points,
                                    min_cube_size=0.3,
                                    max_cube_size=0.7)
   if result["cube_found"]:
       print("面心:", result["face_centers"][0])
   ```

## 立方体尺寸筛选

- `min_cube_size`：允许的最小立方体边长（单位：米），默认 `0.3`。
- `max_cube_size`：允许的最大立方体边长，默认 `0.7`。
- 算法会根据聚类的轴对齐包围盒（AABB）尺寸自动剔除过小或过大的候选物体，以减少噪声与误检。

可根据实际场景调整 `ground_distance_threshold`、`cluster_eps` 等参数，以适配不同的点云密度和噪声水平。

## 可视化说明

- 灰色：原始场景点云
- 红色：检测到的立方体可见面
- 绿色球：各面的中心点
- 坐标轴：摄像机（雷达）坐标系参考

## 后续计划

- 融合 ROS2 话题订阅，实现在线立方体检测
- 引入多帧数据融合，提升稳定性
- 与机器人控制逻辑对接，实现动态抓取/避障

如需反馈或扩展功能，欢迎提交 Issue 或 PR。
