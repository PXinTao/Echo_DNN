from ddn_annealed import AnnealedSelectorConfig, FitnessWeights, DiversityConfig, LayerTemperaturePolicy

class CardiacUltrasoundConfig:
    """心脏超声专用配置类"""
    
    @staticmethod
    def get_default_config():
        """获取心脏超声的默认配置"""
        return AnnealedSelectorConfig(
            weights=FitnessWeights(
                quality=1.0,      # 医学图像质量至关重要
                diversity=0.3,    # 适度多样性，避免模式坍塌
                minority=0.7      # 高度重视罕见心脏疾病
            ),
            diversity=DiversityConfig(
                mode="centroid",  # 使用质心模式计算多样性
                pool_size=8,      # 适中的池化大小用于特征提取
                sample_m=16       # 采样数量
            ),
            temp_policy=LayerTemperaturePolicy(
                init_T=2.0,              # 较高初始温度，医学图像需要更多探索
                min_T=0.005,             # 很低的最小温度，确保高质量细节
                max_T=4.0,               # 较高最大温度，处理复杂病例
                global_decay=1.5e-4,     # 较慢的全局衰减
                first_layer_boost=2.0,   # 早期层高探索，学习整体心脏结构
                last_layer_factor=0.05   # 后期层极高精度，学习细微纹理
            ),
            sa_steps=1,              # 使用1步模拟退火优化
            seed_base=20250827       # 固定随机种子基数
        )
    
    @staticmethod
    def get_rare_disease_config():
        """专门针对罕见心脏疾病的配置"""
        return AnnealedSelectorConfig(
            weights=FitnessWeights(
                quality=0.8,     # 稍微降低质量权重
                diversity=0.2,   # 降低多样性权重  
                minority=1.0     # 最大化少数类权重
            ),
            temp_policy=LayerTemperaturePolicy(
                init_T=3.0,              # 更高初始温度
                min_T=0.1,               # 更高最小温度，保持探索性
                first_layer_boost=3.0,   # 极高的早期探索
                last_layer_factor=0.2    # 适中的后期精度
            ),
            sa_steps=2  # 更多的退火步骤
        )
    
    @staticmethod 
    def get_high_quality_config():
        """专门针对高质量生成的配置"""
        return AnnealedSelectorConfig(
            weights=FitnessWeights(
                quality=1.5,     # 超高质量权重
                diversity=0.1,   # 最小多样性权重
                minority=0.4     # 适中的少数类权重
            ),
            temp_policy=LayerTemperaturePolicy(
                init_T=1.0,              # 较低初始温度
                min_T=0.001,             # 极低最小温度
                max_T=2.0,               # 适中最大温度
                last_layer_factor=0.01   # 极高后期精度
            ),
            sa_steps=0  # 不使用模拟退火，直接softmax选择
        )