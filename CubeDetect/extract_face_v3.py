"""
Cube Detection V3 - 针对稀疏激光雷达点云优化的立方体检测算法

主要改进：
1. 使用更保守的地面移除 - 只移除明确的水平地面
2. 基于OBB（有向包围盒）的立方体检测 - 更适合任意朝向的立方体
3. 简化的几何验证 - 不依赖复杂的平面检测
4. 多尺度聚类 - 自适应调整聚类参数
"""

import open3d as o3d
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CubeDetectorConfigV3:
    """V3版本的配置参数"""
    # 范围过滤
    min_range: float = 0.3
    max_range: float = 5.0
    
    # 地面移除 - 基于高度的简单方法
    ground_height_threshold: float = 0.10  # 低于此高度的点视为地面
    use_ransac_ground: bool = False  # 是否使用RANSAC（默认关闭）
    
    # 聚类参数
    cluster_eps: float = 0.08  # 显著减小，对近距离目标更敏感
    cluster_min_points: int = 15  # 减少最小点数要求
    
    # 立方体尺寸约束
    min_cube_size: float = 0.15  # 最小边长
    max_cube_size: float = 0.8   # 最大边长
    target_cube_size: float = 0.5  # 目标边长（用于评分）
    
    # 形状验证
    max_aspect_ratio: float = 2.5  # 最大长宽比
    min_3d_ratio: float = 0.3  # 最小的第三维度比例（防止平板）
    
    # 点密度
    min_density_score: float = 0.0  # 最小点密度分数


class CubeDetectorV3:
    """V3版本的立方体检测器"""
    
    def __init__(self, config: CubeDetectorConfigV3 = None):
        self.config = config if config else CubeDetectorConfigV3()
    
    def detect(self, points: np.ndarray) -> List[dict]:
        """
        主检测函数
        
        Args:
            points: Nx3 numpy数组
            
        Returns:
            候选立方体列表
        """
        if points is None or len(points) < 10:
            return []
        
        # 确保是Nx3
        if points.shape[1] > 3:
            points = points[:, :3]
        
        # Step 1: 范围过滤
        points = self._filter_by_range(points)
        if len(points) < self.config.cluster_min_points:
            return []
        
        # Step 2: 地面移除
        points = self._remove_ground(points)
        if len(points) < self.config.cluster_min_points:
            return []
        
        # Step 3: 聚类
        clusters = self._cluster_points(points)
        if not clusters:
            return []
        
        # Step 4: 对每个簇进行立方体评估
        candidates = []
        for cluster_points in clusters:
            result = self._evaluate_cluster(cluster_points)
            if result is not None:
                candidates.append(result)
        
        # 按分数排序
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates
    
    def _filter_by_range(self, points: np.ndarray) -> np.ndarray:
        """范围过滤"""
        dists = np.linalg.norm(points, axis=1)
        mask = (dists >= self.config.min_range) & (dists <= self.config.max_range)
        return points[mask]
    
    def _remove_ground(self, points: np.ndarray) -> np.ndarray:
        """
        简单的地面移除 - 基于高度阈值
        假设雷达安装在一定高度，地面点的z值较低
        """
        if self.config.use_ransac_ground:
            return self._remove_ground_ransac(points)
        
        # 简单方法：找到最低的z值，移除接近它的点
        z_values = points[:, 2]
        z_min = np.percentile(z_values, 5)  # 使用5%分位数作为地面参考
        
        # 保留高于地面阈值的点
        mask = z_values > (z_min + self.config.ground_height_threshold)
        return points[mask]
    
    def _remove_ground_ransac(self, points: np.ndarray) -> np.ndarray:
        """使用RANSAC的地面移除（可选）"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.05,
                ransac_n=3,
                num_iterations=200
            )
            # 检查是否是水平面（法向量接近垂直）
            normal = np.array(plane_model[:3])
            if abs(normal[2]) > 0.8:  # 法向量接近z轴
                return np.asarray(pcd.select_by_index(inliers, invert=True).points)
        except Exception:
            pass
        
        return points
    
    def _cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        """
        自适应聚类 - 尝试多个eps值
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 尝试不同的eps值
        eps_values = [
            self.config.cluster_eps,
            self.config.cluster_eps * 1.5,
            self.config.cluster_eps * 2.0,
        ]
        
        best_clusters = []
        best_valid_count = 0
        
        for eps in eps_values:
            labels = np.array(pcd.cluster_dbscan(
                eps=eps,
                min_points=self.config.cluster_min_points,
                print_progress=False
            ))
            
            if labels.size == 0 or labels.max() < 0:
                continue
            
            clusters = []
            for i in range(labels.max() + 1):
                cluster_mask = labels == i
                cluster_points = points[cluster_mask]
                if len(cluster_points) >= self.config.cluster_min_points:
                    clusters.append(cluster_points)
            
            # 统计可能是立方体的簇数量
            valid_count = sum(1 for c in clusters if self._quick_check(c))
            
            if valid_count > best_valid_count:
                best_valid_count = valid_count
                best_clusters = clusters
        
        return best_clusters
    
    def _quick_check(self, points: np.ndarray) -> bool:
        """快速检查一个簇是否可能是立方体"""
        if len(points) < self.config.cluster_min_points:
            return False
        
        # 计算包围盒
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        extent = max_pt - min_pt
        
        max_dim = np.max(extent)
        min_dim = np.min(extent)
        
        # 基本尺寸检查
        if max_dim > self.config.max_cube_size:
            return False
        if max_dim < self.config.min_cube_size:
            return False
        
        return True
    
    def _evaluate_cluster(self, points: np.ndarray) -> Optional[dict]:
        """
        评估一个点云簇是否是立方体
        
        使用OBB（有向包围盒）来更好地拟合立方体
        """
        if len(points) < self.config.cluster_min_points:
            return None
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 计算AABB和OBB
        aabb = pcd.get_axis_aligned_bounding_box()
        
        try:
            obb = pcd.get_oriented_bounding_box()
            # 使用OBB的尺寸
            extent = obb.extent
            center = obb.center
        except Exception:
            # 退回到AABB
            extent = aabb.get_extent()
            center = aabb.get_center()
        
        # 排序尺寸
        sorted_extent = np.sort(extent)
        min_dim, mid_dim, max_dim = sorted_extent
        
        # === 尺寸验证 ===
        
        # 1. 绝对尺寸限制
        if max_dim > self.config.max_cube_size:
            return None
        if max_dim < self.config.min_cube_size:
            return None
        
        # 2. 长宽比检查（不能太扁长）
        if mid_dim > 0:
            aspect_ratio = max_dim / mid_dim
            if aspect_ratio > self.config.max_aspect_ratio:
                return None
        
        # 3. 三维性检查（不能是平板）
        if max_dim > 0:
            thickness_ratio = min_dim / max_dim
            if thickness_ratio < self.config.min_3d_ratio:
                # 可能只看到了立方体的一个面，这种情况下放宽限制
                # 但至少要有一定的厚度
                if min_dim < 0.05:
                    return None
        
        # === 评分 ===
        score = self._calculate_score(points, extent, center)
        
        if score <= 0:
            return None
        
        # 计算距离
        distance = np.linalg.norm(center)
        
        # 估算面中心（简化版，不依赖平面检测）
        face_centers = self._estimate_face_centers(points, center, extent)
        
        return {
            'cube_center': center.tolist(),
            'face_centers': face_centers,
            'extent': extent.tolist(),
            'num_points': len(points),
            'distance': distance,
            'score': score,
            'confidence': self._get_confidence_label(score, len(points), extent)
        }
    
    def _calculate_score(self, points: np.ndarray, extent: np.ndarray, 
                         center: np.ndarray) -> float:
        """计算立方体可能性分数"""
        score = 100.0
        
        sorted_extent = np.sort(extent)
        min_dim, mid_dim, max_dim = sorted_extent
        
        # 1. 尺寸接近目标的奖励（高斯权重）
        avg_dim = np.mean(extent)
        size_diff = abs(avg_dim - self.config.target_cube_size)
        # 使用高斯函数，sigma=0.15，尺寸越接近目标分数越高
        size_score = 80 * np.exp(-(size_diff ** 2) / (2 * 0.15 ** 2))
        score += size_score
        
        # 2. 立方体形状奖励（三边接近相等）
        if max_dim > 0:
            # 计算变异系数（CV），完美立方体CV=0
            cv = np.std(extent) / np.mean(extent)
            # CV < 0.3 是比较好的立方体
            if cv < 0.2:
                score += 60  # 非常接近立方体
            elif cv < 0.4:
                score += 40
            elif cv < 0.6:
                score += 20
            # CV > 0.6 的不给形状分
        
        # 3. 点数奖励（使用对数缩放）
        num_points = len(points)
        point_score = min(50, 15 * np.log10(max(num_points, 10)))
        score += point_score
        
        # 4. 距离惩罚（远处目标降低可信度）
        distance = np.linalg.norm(center)
        if distance > 2.0:
            score -= (distance - 2.0) * 20
        
        # 5. 点云紧凑度奖励
        volume = np.prod(extent)
        if volume > 0.001:  # 避免除零
            density = num_points / (volume * 1000)  # 归一化
            if density > 0.5:
                score += 25
            elif density > 0.2:
                score += 10
        
        # 6. 额外检查：使用PCA分析点云分布
        pca_score = self._pca_cube_score(points)
        score += pca_score * 30
        
        return score
    
    def _pca_cube_score(self, points: np.ndarray) -> float:
        """
        使用PCA分析点云是否呈立方体分布
        返回0-1的分数
        """
        if len(points) < 10:
            return 0.0
        
        try:
            # 中心化
            centered = points - np.mean(points, axis=0)
            # 计算协方差矩阵
            cov = np.cov(centered.T)
            # 特征值
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]  # 降序
            
            # 对于立方体，三个特征值应该接近相等
            if eigenvalues[0] > 0:
                # 计算特征值比例
                ratio1 = eigenvalues[1] / eigenvalues[0]
                ratio2 = eigenvalues[2] / eigenvalues[0]
                
                # 理想立方体: ratio1 ≈ 1, ratio2 ≈ 1
                # 平板: ratio2 << 1
                # 细长物体: ratio1 << 1, ratio2 << 1
                
                if ratio1 > 0.3 and ratio2 > 0.1:
                    # 三维分布较均匀
                    return min(1.0, (ratio1 + ratio2) / 2)
                elif ratio1 > 0.5 and ratio2 > 0.05:
                    # 可能只看到两个面
                    return 0.5
            
        except Exception:
            pass
        
        return 0.3  # 默认分数
    
    def _estimate_face_centers(self, points: np.ndarray, center: np.ndarray, 
                                extent: np.ndarray) -> List[List[float]]:
        """
        使用RANSAC平面分割来精确估算可见面的中心
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        face_centers = []
        face_infos = []  # 存储面信息用于排序
        temp_pcd = pcd
        
        # 尝试提取最多3个平面
        for _ in range(3):
            if len(temp_pcd.points) < 10:
                break
            
            try:
                plane_model, inliers = temp_pcd.segment_plane(
                    distance_threshold=0.03,  # 更严格的阈值
                    ransac_n=3,
                    num_iterations=100
                )
                
                if len(inliers) < 8:
                    break
                
                # 获取面的点云
                face_pcd = temp_pcd.select_by_index(inliers)
                face_points = np.asarray(face_pcd.points)
                
                # 计算面中心（使用中位数更鲁棒）
                face_center = np.median(face_points, axis=0)
                
                # 验证面中心是否合理（应该在点云边界附近）
                min_pt = np.min(points, axis=0)
                max_pt = np.max(points, axis=0)
                
                # 检查面中心是否在合理范围内
                margin = 0.1  # 10cm的余量
                if np.all(face_center >= min_pt - margin) and np.all(face_center <= max_pt + margin):
                    face_infos.append({
                        'center': face_center,
                        'count': len(inliers),
                        'normal': np.array(plane_model[:3])
                    })
                
                # 移除已处理的点
                temp_pcd = temp_pcd.select_by_index(inliers, invert=True)
                
            except Exception:
                break
        
        # 按点数排序，返回最大的几个面的中心
        face_infos.sort(key=lambda x: x['count'], reverse=True)
        face_centers = [f['center'].tolist() for f in face_infos[:3]]
        
        return face_centers
    
    def _get_confidence_label(self, score: float, num_points: int, 
                              extent: np.ndarray) -> str:
        """获取置信度标签"""
        sorted_extent = np.sort(extent)
        cube_ratio = sorted_extent[0] / sorted_extent[2] if sorted_extent[2] > 0 else 0
        
        # 计算变异系数
        cv = np.std(extent) / np.mean(extent) if np.mean(extent) > 0 else 1.0
        
        if score > 200 and num_points > 60 and cv < 0.4:
            return "HIGH"
        elif score > 150 and num_points > 30:
            return "MEDIUM"
        elif score > 100:
            return "LOW"
        else:
            return "VERY_LOW"


def detect_cube_v3(points: np.ndarray, config: CubeDetectorConfigV3 = None) -> dict:
    """
    V3检测接口 - 与V2兼容的接口
    """
    if config is None:
        config = CubeDetectorConfigV3()
    
    detector = CubeDetectorV3(config)
    candidates = detector.detect(points)
    
    return {
        'cube_found': len(candidates) > 0,
        'candidates': candidates
    }


# 兼容性别名
detect_cube_v3_simple = detect_cube_v3
