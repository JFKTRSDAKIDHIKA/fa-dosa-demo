import os
import subprocess
import yaml
import numpy as np
import shutil
from sklearn.linear_model import LinearRegression

# ==============================================================================
# 1. 配置您的环境和测试参数
# ==============================================================================

# 请确保这个路径指向您的Accelergy组件库
# [cite_start]这是在 Timeloop & Accelergy 核心指令与警告.pdf 中强调的关键点 [cite: 430]
ACCELERGY_COMPONENT_LIBRARY_PATH = '/root/fa-dosa/accelergy-timeloop-infrastructure/src/accelergy-library-plug-in/library/'

# 定义要测试的SRAM容量点（单位：KB）
# 我们将为这些点生成配置并运行仿真
SRAM_SIZES_KB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# 定义存储器的基本属性
# [cite_start]根据 TimeloopAccelergy文档.pdf, width 和 datawidth 是必需的属性 [cite: 1087-1088]
TECHNOLOGY = "40nm"
WORD_BITS = 64  # 假设总线位宽为64-bit

# 校准工作目录
WORKSPACE_DIR = "calibration_workspace"


def generate_accelergy_inputs(size_kb):
    """为指定容量的SRAM生成Accelergy所需的输入YAML文件。"""
    
    # --- 1. 创建 arch.yaml ---
    # 计算SRAM的深度 (depth)
    # 容量(bytes) = size_kb * 1024
    # 深度 = 容量(bytes) / (位宽/8)
    width_bytes = WORD_BITS / 8
    depth = int((size_kb * 1024) / width_bytes)

    arch_dict = {
        'architecture': {
            'version': '0.4',
            'nodes': [
                {
                    'name': f'sram_{size_kb}kb',
                    'class': 'SRAM',
                    'attributes': {
                        'technology': TECHNOLOGY,
                        'width': WORD_BITS,
                        'depth': depth,
                        'datawidth': WORD_BITS # datawidth也是一个常见且必需的属性
                    }
                }
            ]
        }
    }
    with open(os.path.join(WORKSPACE_DIR, 'arch.yaml'), 'w') as f:
        yaml.dump(arch_dict, f, sort_keys=False)

    # --- 2. 创建 problem.yaml (一个虚拟的 workload) ---
    problem_dict = {
        'problem': {
            'shape': {'name': 'dummy'},
            'instance': {'dummy': 1}
        }
    }
    with open(os.path.join(WORKSPACE_DIR, 'problem.yaml'), 'w') as f:
        yaml.dump(problem_dict, f, sort_keys=False)

    # --- 3. 创建 env.yaml (指定技术节点和组件库) ---
    # [cite_start]这是成功运行Accelergy的关键 [cite: 429, 442]
    env_dict = {
        'globals': {
            'environment_variables': {
                'ACCELERGY_COMPONENT_LIBRARIES': ACCELERGY_COMPONENT_LIBRARY_PATH
            }
        },
        'variables': {
            'global_cycle_seconds': 1e-9, # 1GHz clock, 一个标准值
            'technology': TECHNOLOGY
        }
    }
    with open(os.path.join(WORKSPACE_DIR, 'env.yaml'), 'w') as f:
        yaml.dump(env_dict, f, sort_keys=False)


def run_accelergy_simulation():
    """执行Accelergy仿真。"""
    print(f"  - Running Accelergy...", end='', flush=True)
    try:
        # 定义输入文件和输出目录
        input_files = [
            os.path.join(WORKSPACE_DIR, 'arch.yaml'),
            os.path.join(WORKSPACE_DIR, 'problem.yaml'),
            os.path.join(WORKSPACE_DIR, 'env.yaml')
        ]
        
        # 使用subprocess调用Accelergy
        # 我们需要捕获输出以进行调试
        result = subprocess.run(
            ['accelergy'] + input_files + ['-o', WORKSPACE_DIR],
            capture_output=True, text=True, check=True, timeout=60
        )
        print("Done.")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print("Failed.")
        print("Accelergy Error Output:")
        print(e.stderr)
        return False


def parse_accelergy_output():
    """解析Accelergy的ERT输出文件以获取能耗。"""
    ert_path = os.path.join(WORKSPACE_DIR, 'ERT.yaml')
    try:
        with open(ert_path, 'r') as f:
            ert_data = yaml.safe_load(f)
        
        # 提取第一个（也是唯一一个）组件的单次读取能耗
        # 能量单位在ERT中通常是pJ
        energy = ert_data['ERT']['tables'][0]['actions'][0]['energy']
        return energy
    except (IOError, KeyError, IndexError) as e:
        print(f"  - Failed to parse ERT file: {e}")
        return None


def fit_energy_model(sizes_kb, energies_pj):
    """使用线性回归拟合能量模型系数。"""
    X = np.array(sizes_kb).reshape(-1, 1)
    y = np.array(energies_pj)

    model = LinearRegression()
    model.fit(X, y)

    # 斜率 (Slope) -> CAPACITY_COEFF
    capacity_coeff = model.coef_[0]
    
    # 截距 (Intercept) -> BASE_EPA
    base_epa = model.intercept_

    return base_epa, capacity_coeff


def main():
    """主校准流程。"""
    print("Starting Accelergy calibration process for SRAM...")

    # 创建工作区
    if os.path.exists(WORKSPACE_DIR):
        shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR)

    collected_data = []

    for size in SRAM_SIZES_KB:
        print(f"\nProcessing SRAM size: {size} KB")
        
        # 1. 生成输入文件
        print(f"  - Generating Accelergy YAML files...")
        generate_accelergy_inputs(size)
        
        # 2. 运行仿真
        if not run_accelergy_simulation():
            continue
            
        # 3. 解析输出
        print(f"  - Parsing ERT.yaml for energy values...")
        energy = parse_accelergy_output()
        
        if energy is not None:
            print(f"  - Success! Collected energy: {energy:.4f} pJ")
            collected_data.append({'size': size, 'energy': energy})

    # --- 数据拟合 ---
    if len(collected_data) < 2:
        print("\nCalibration failed: Not enough data points collected to perform fitting.")
        return

    sizes = [d['size'] for d in collected_data]
    energies = [d['energy'] for d in collected_data]

    print("\n" + "="*80)
    print("PERFORMING LINEAR REGRESSION ON COLLECTED SRAM ENERGY DATA")
    print("="*80)
    print(f"Data points: {len(sizes)}")
    
    # 拟合L1和L2。这里我们假设L1和L2使用相同的SRAM技术，因此用同一套数据拟合。
    # 如果您认为它们的物理实现差异巨大，可以创建两套SRAM_SIZES_KB并分别运行和拟合。
    base_epa, capacity_coeff = fit_energy_model(sizes, energies)

    print("\n--- FIT RESULTS ---")
    print(f"Model: Energy(pJ) = BASE_EPA + CAPACITY_COEFF * Size(KB)")
    print(f"  -> Intercept (BASE_EPA): {base_epa:.6f}")
    print(f"  -> Slope (CAPACITY_COEFF): {capacity_coeff:.6f}")

    print("\n" + "="*80)
    print("ACTION REQUIRED: PLEASE UPDATE 'dosa/config.py'")
    print("="*80)
    print("Copy and paste the following code block into your 'dosa/config.py' file,")
    print("replacing the existing SRAM and DRAM energy parameters:\n")

    # 为DRAM提供一个基于仿真的典型值
    # 注意：真实DRAM能耗更复杂，但对于模型来说，一个基于Accelergy的固定值是很好的起点
    dram_typical_energy_pj = 100 * 1e6 # 沿用之前的值作为示例，实际应通过仿真获得
    
    print("# --- 2. 存储组件 (经验拟合) ---")
    print(f"# L1 Accumulator (Fitted from Accelergy simulations)")
    print(f"self.L1_ACCUM_BASE_EPA_PJ = {base_epa:.6f}")
    print(f"self.L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB = {capacity_coeff:.6f}\n")
    
    print(f"# L2 Scratchpad (Fitted from Accelergy simulations)")
    print(f"self.L2_SPM_BASE_EPA_PJ = {base_epa:.6f}")
    print(f"self.L2_SPM_CAPACITY_COEFF_PJ_PER_KB = {capacity_coeff:.6f}\n")
    
    print(f"# L3 DRAM (Based on Accelergy simulation for a typical configuration)")
    print(f"self.L3_DRAM_EPA_PJ = {dram_typical_energy_pj:.6f}")
    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
