import torch
import torch.nn as nn

class Config:
    """
    fa-dosa框架核心硬件规约中心 - 单例模式
    
    本类集中管理所有物理常数、硬件约束和能耗模型参数，
    为高精度性能与能耗分析提供统一的物理基础。
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if Config._initialized:
            return
        Config._initialized = True
        
        # ========== 基础硬件参数 ==========
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.BYTES_PER_ELEMENT = 4  # float32
        self.CLOCK_FREQUENCY_MHZ = 1000  # 1GHz
        
        # ========== 多级存储层次定义 ==========
        # 索引 i=0,1,2,3 对应 L0_Registers, L1_Accumulator, L2_Scratchpad, L3_DRAM
        self.MEMORY_HIERARCHY = [
            {
                'index': 0,
                'name': 'L0_Registers', 
                'type': 'buffer', 
                'description': 'PE内部寄存器，暂存操作数'
            },
            {
                'index': 1,
                'name': 'L1_Accumulator', 
                'type': 'buffer', 
                'description': '累加器缓存，存储输出部分和'
            },
            {
                'index': 2,
                'name': 'L2_Scratchpad', 
                'type': 'buffer', 
                'description': '片上共享缓存，层级间数据交换'
            },
            {
                'index': 3,
                'name': 'L3_DRAM', 
                'type': 'dram', 
                'description': '主存，存储完整模型权重和数据'
            }
        ]
        
        # ========== 问题维度定义 ==========
        # 7个核心维度：R,S,P,Q,C,K,N
        self.PROBLEM_DIMENSIONS = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
        
        # ========== 张量维度映射 ==========
        # 定义每种张量关联的维度集合 D_t
        self.TENSOR_DIMENSIONS = {
            'W': {'K', 'C', 'R', 'S'},      # 权重张量 D_W
            'I': {'N', 'C', 'P', 'Q'},      # 输入张量 D_I (简化版，实际包含R,S)
            'O': {'N', 'K', 'P', 'Q'}       # 输出张量 D_O
        }
        self.D_ALL = {'R', 'S', 'P', 'Q', 'C', 'K', 'N'}  # 所有维度
        
        # ========== 存储矩阵 B_i,t ==========
        # 定义哪种张量(t)可以存储在哪一级内存(i)
        # 1=允许存储, 0=不允许存储
        self.STORAGE_MATRIX = {
            0: {'W': 1, 'I': 1, 'O': 0},  # L0_Registers: 权重+输入，无输出
            1: {'W': 0, 'I': 0, 'O': 1},  # L1_Accumulator: 仅输出
            2: {'W': 1, 'I': 1, 'O': 1},  # L2_Scratchpad: 全部张量
            3: {'W': 1, 'I': 1, 'O': 1}   # L3_DRAM: 全部张量
        }
        
        # ========== 能耗模型参数 (单位: pJ) ==========
        # MAC运算能耗
        self.PE_MAC_EPA_PJ = 0.845
        
        # L0寄存器访问能耗
        self.L0_REG_BASE_EPA_PJ = 9.4394
        
        # L1累加器能耗模型: EPA_L1 = Base + Coeff * (Capacity_KB / sqrt(Num_PE))
        self.L1_ACCUM_BASE_EPA_PJ = 1.94
        self.L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB = 0.1005
        
        # L2缓存能耗模型: EPA_L2 = Base + Coeff * Capacity_KB
        self.L2_SPM_BASE_EPA_PJ = 31.0663
        self.L2_SPM_CAPACITY_COEFF_PJ_PER_KB = 0.3
        
        # L3 DRAM访问能耗
        self.L3_DRAM_EPA_PJ = 128.0
        
        # ========== 优化目标与损失函数权重 ==========
        self.LOSS_STRATEGY = 'pure_edp'
        # Area penalty is disabled by default so that the benefit of larger
        # hardware configurations can be observed. Set ``AREA_WEIGHT`` to a
        # non-zero value to re-enable the penalty.
        # Use a small positive value by default to encourage exploring larger
        # hardware when needed.
        self.AREA_WEIGHT = 0.1
        self.COMPATIBILITY_PENALTY_WEIGHT = 100.0
        # Weight for buffer mismatch penalty
        self.MISMATCH_PENALTY_WEIGHT = 0.1
        
        # ========== 面积预算约束配置 ==========
        # 是否启用面积预算约束
        self.ENABLE_AREA_BUDGET = False
        # 面积预算值 (mm²) - 设置为None表示无预算限制
        self.AREA_BUDGET_MM2 = None
        # 预算容忍区间 (百分比) - 在预算±tolerance范围内不施加惩罚
        self.AREA_BUDGET_TOLERANCE = 0.1  # 10%
        # 面积预算惩罚权重 - 控制惩罚强度
        self.AREA_BUDGET_PENALTY_WEIGHT = 1.0
        # 面积预算惩罚策略: 'quadratic', 'huber', 'exponential', 'linear'
        self.AREA_BUDGET_PENALTY_STRATEGY = 'quadratic'
        # Huber损失的delta参数 (仅在strategy='huber'时使用)
        self.AREA_BUDGET_HUBER_DELTA = 0.5
        # 权重调度参数 - 惩罚权重随训练步数增加
        self.AREA_BUDGET_WEIGHT_SCHEDULE = {
            'enable': True,
            'initial_weight': 0.1,  # 初始权重
            'final_weight': 2.0,    # 最终权重
            'warmup_steps': 100,    # 预热步数
            'schedule_type': 'linear'  # 'linear', 'exponential'
        }
        
        # 场景预设配置
        self.SCENARIO_PRESETS = {
            'edge': {
                'area_budget_mm2': 5.0,
                'tolerance': 0.05,  # 5% - 更严格
                'penalty_weight': 2.0,  # 更强惩罚
                'penalty_strategy': 'quadratic'
            },
            'cloud': {
                'area_budget_mm2': 50.0,
                'tolerance': 0.15,  # 15% - 更宽松
                'penalty_weight': 0.5,  # 较弱惩罚
                'penalty_strategy': 'huber'
            },
            'mobile': {
                'area_budget_mm2': 2.0,
                'tolerance': 0.03,  # 3% - 极严格
                'penalty_weight': 5.0,  # 极强惩罚
                'penalty_strategy': 'exponential'
            }
        }
        
        # Consolidated loss weights for easy tuning
        self.LOSS_WEIGHTS = {
            'area_weight': self.AREA_WEIGHT,
            'mismatch_penalty_weight': self.MISMATCH_PENALTY_WEIGHT,
            'compatibility_penalty_weight': self.COMPATIBILITY_PENALTY_WEIGHT,
            'edp_weight': 1.0,
            'area_budget_penalty_weight': self.AREA_BUDGET_PENALTY_WEIGHT
        }
        
        # ========== 面积模型参数 (单位: mm²) ==========
        self.AREA_BASE_MM2 = 1.0                # 芯片基础面积
        self.AREA_PER_PE_MM2 = 0.01             # 每个PE的面积成本
        self.AREA_PER_KB_L1_MM2 = 0.1           # L1缓存每KB面积成本
        self.AREA_PER_KB_L2_MM2 = 0.05          # L2缓存每KB面积成本
        
        # ========== 带宽模型参数 ==========
        self.DRAM_BANDWIDTH_GB_S = 100  # 对应8 words/cycle @ 1GHz
        
        # ========== 数据供给通路图 ==========
        # 定义每个存储层级中，各类张量数据的合法供给来源
        # 'PE' 表示由计算单元直接产生
        self.DATA_SUPPLY_MAP = {
            'L0_Registers': {
                'Input': 'L2_Scratchpad',   # L0的Input数据来自L2
                'Weight': 'L2_Scratchpad',  # L0的Weight数据也来自L2
                'Output': 'PE'              # Output由PE计算单元产生，无需填充
            },
            'L1_Accumulator': {
                'Output': 'L0_Registers'    # L1的Output数据来自L0的写回
            },
            'L2_Scratchpad': {
                'Input': 'L3_DRAM',
                'Weight': 'L3_DRAM',
                'Output': 'L1_Accumulator'  # L2的Output数据来自L1的写回
            },
            'L3_DRAM': {
                # DRAM是最高层级，其数据被认为是"凭空而来"，主要接受来自下层的写回
                'Output': 'L2_Scratchpad'
            }
        }
        
        # ========== 日志与可观测性配置 ==========
        self.MINIMAL_CONSOLE = True          # 控制台仅输出最少必要信息
        self.LOG_INTERMEDIATE = True         # 记录中间过程到文件
        self.LOG_DIR = "output"              # 日志输出目录
        self.LOG_TRIAL_INTERVAL = 50         # 控制台输出试验结果的间隔
        self.LOG_VALIDATION_INTERVAL = 50    # 保存验证配置的间隔

        # ========== 硬件搜索行为配置 ==========
        # If True, hardware parameters are reset to the minimal configuration
        # derived from the current mapping before Phase B. If False, the
        # minimal hardware only serves as a lower bound constraint.
        self.RESET_TO_MIN_HW = False
        # Whether to enforce minimal hardware as a hard lower bound during
        # the search. Set to ``False`` to temporarily remove this constraint
        # and allow more aggressive exploration of the hardware space.
        self.APPLY_MIN_HW_BOUNDS = True
    
    @classmethod
    def get_instance(cls):
        """获取Config单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_memory_level_name(self, index: int) -> str:
        """根据索引获取内存层级名称"""
        if 0 <= index < len(self.MEMORY_HIERARCHY):
            return self.MEMORY_HIERARCHY[index]['name']
        raise ValueError(f"Invalid memory level index: {index}")
    
    def get_tensor_dimensions(self, tensor_type: str) -> set:
        """获取指定张量类型的维度集合"""
        if tensor_type in self.TENSOR_DIMENSIONS:
            return self.TENSOR_DIMENSIONS[tensor_type]
        raise ValueError(f"Unknown tensor type: {tensor_type}")
    
    def can_store_tensor(self, level_index: int, tensor_type: str) -> bool:
        """检查指定层级是否可以存储指定类型的张量"""
        if level_index in self.STORAGE_MATRIX and tensor_type in self.STORAGE_MATRIX[level_index]:
            return bool(self.STORAGE_MATRIX[level_index][tensor_type])
        return False
    
    def apply_scenario_preset(self, scenario: str):
        """应用场景预设配置
        
        Args:
            scenario: 场景名称 ('edge', 'cloud', 'mobile')
        """
        if scenario not in self.SCENARIO_PRESETS:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(self.SCENARIO_PRESETS.keys())}")
        
        preset = self.SCENARIO_PRESETS[scenario]
        
        # 启用面积预算并应用预设参数
        self.ENABLE_AREA_BUDGET = True
        self.AREA_BUDGET_MM2 = preset['area_budget_mm2']
        self.AREA_BUDGET_TOLERANCE = preset['tolerance']
        self.AREA_BUDGET_PENALTY_WEIGHT = preset['penalty_weight']
        self.AREA_BUDGET_PENALTY_STRATEGY = preset['penalty_strategy']
        
        # 更新loss weights
        self.LOSS_WEIGHTS['area_budget_penalty_weight'] = self.AREA_BUDGET_PENALTY_WEIGHT
        
        print(f"[CONFIG] 已应用{scenario}场景预设: 预算={self.AREA_BUDGET_MM2}mm², 容忍度={self.AREA_BUDGET_TOLERANCE*100}%, 惩罚权重={self.AREA_BUDGET_PENALTY_WEIGHT}")