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
# 这是在 Timeloop & Accelergy 核心指令与警告.pdf 中强调的关键点
ACCELERGY_COMPONENT_LIBRARY_PATH = '/root/fa-dosa/accelergy-timeloop-infrastructure/src/accelergy-library-plug-in/library/'

# 定义要测试的SRAM容量点（单位：KB）
SRAM_SIZES_KB = [64, 128, 256, 512, 1024]

# 定义存储器的基本属性
# 根据 TimeloopAccelergy文档.pdf, width 和 datawidth 是必需的属性
TECHNOLOGY = "40nm"  # 这是一个字符串，但在YAML中需特殊处理
WORD_BITS = 64      # 假设总线位宽为64-bit

# 校准工作目录
WORKSPACE_DIR = "calibration_workspace"


def generate_accelergy_inputs(size_kb):
    """为指定容量的SRAM生成Accelergy所需的输入YAML文件。"""
    width_bytes = WORD_BITS / 8
    depth = int((size_kb * 1024) / width_bytes)
    component_name = f'sram_{size_kb}kb'

    # --- 1. 创建 arch.yaml ---
    arch_dict = {
        'architecture': {
            'version': '0.4',
            'local': [
                {
                    'name': component_name,
                    'class': 'sram',
                    'attributes': {
                        'technology': TECHNOLOGY,
                        'width': WORD_BITS,
                        'depth': depth,
                        'datawidth': WORD_BITS
                    }
                }
            ]
        }
    }
    with open(os.path.join(WORKSPACE_DIR, 'arch.yaml'), 'w') as f:
        yaml.dump(arch_dict, f, sort_keys=False, default_flow_style=False)

    # --- 2. 创建 problem.yaml (最终正确版本) ---
    problem_and_actions_dict = {
        'problem': {
            'version': '0.4',
            'shape': {'name': 'dummy'},
            'instance': {'dummy': 1}
        },
        'action_counts': {
            'version': '0.4',
            'local': [
                {
                    'name': component_name,
                    # 关键修正：这里的 action_counts 必须是一个动作列表 (List of Actions)
                    'action_counts': [
                        {
                            'name': 'read',   # 每个动作都是一个字典
                            'counts': 1      # 包含 'name' 和 'counts' 键
                        }
                    ]
                }
            ]
        }
    }
    with open(os.path.join(WORKSPACE_DIR, 'problem.yaml'), 'w') as f:
        yaml.dump(problem_and_actions_dict, f, sort_keys=False, default_flow_style=False)

    # --- 3. 创建 env.yaml ---
    env_dict = {
        'globals': {
            'environment_variables': {
                'ACCELERGY_COMPONENT_LIBRARIES': ACCELERGY_COMPONENT_LIBRARY_PATH
            }
        },
        'variables': {
            'global_cycle_seconds': 1e-9,
            'technology': TECHNOLOGY
        }
    }
    with open(os.path.join(WORKSPACE_DIR, 'env.yaml'), 'w') as f:
        yaml.dump(env_dict, f, sort_keys=False, default_flow_style=False)

def run_accelergy_simulation():
    """执行Accelergy仿真，并在运行前修复YAML语法问题。"""
    print(f"  - Running Accelergy...", end='', flush=True)

    arch_path = os.path.join(WORKSPACE_DIR, 'arch.yaml')
    env_path = os.path.join(WORKSPACE_DIR, 'env.yaml')

    # 关键修正2: 为 'technology: 40nm' 这样的值加上单引号，确保Accelergy将其识别为字符串
    # 使用sed命令是一个非常可靠的跨平台方法
    fix_command = [
        'sed', '-i', f"s/technology: {TECHNOLOGY}/technology: '{TECHNOLOGY}'/g",
        arch_path, env_path
    ]
    try:
        subprocess.run(fix_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Failed to fix YAML files with sed.")
        print(e.stderr)
        return False

    try:
        input_files = [
            arch_path,
            os.path.join(WORKSPACE_DIR, 'problem.yaml'),
            env_path
        ]
        
        # 调试技巧：始终使用 -v (verbose) 标志！
        # -o 指定输出目录, '.' 表示当前目录
        command = ['accelergy'] + input_files + ['-o', WORKSPACE_DIR, '-v']

        result = subprocess.run(
            command,
            capture_output=True, text=True, check=True, timeout=120
        )
        print("Done.")
        
        # 调试技巧: 如果你想查看Accelergy的详细输出(即使是成功时)，取消下面这几行的注释
        # if result.stdout:
        #     print("  - Accelergy Verbose Output:")
        #     # 打印stdout可以帮助你看到它是否发出了任何警告
        #     print("="*10 + " STDOUT " + "="*10)
        #     print(result.stdout)
        #     print("="*10 + " END STDOUT " + "="*10)

        return True
    except subprocess.CalledProcessError as e:
        print("Failed.")
        print("\n" + "="*20 + " ACCELERGY ERROR " + "="*20)
        print("Accelergy returned a non-zero exit code. This means it crashed or found a fatal error.")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        print("="*50)
        return False
    except subprocess.TimeoutExpired:
        print("Failed (Timeout).")
        return False


def parse_accelergy_output():
    """解析Accelergy的ERT输出文件以获取能耗（增加了健壮性）。"""
    ert_path = os.path.join(WORKSPACE_DIR, 'output', 'ERT.yaml')
    # Accelergy 0.4+ 默认会把输出放在 -o 指定目录下的 output/ 子目录中
    if not os.path.exists(ert_path):
        ert_path = os.path.join(WORKSPACE_DIR, 'ERT.yaml') # 兼容旧版
        if not os.path.exists(ert_path):
            print("\n  [DIAGNOSIS] ERT.yaml not found in workspace or workspace/output.")
            return None

    try:
        with open(ert_path, 'r') as f:
            ert_data = yaml.safe_load(f)
        
        # 健壮性检查 1: ERT.yaml 的基本结构是否正确？
        if 'ERT' not in ert_data or 'tables' not in ert_data['ERT'] or not ert_data['ERT']['tables']:
            print("\n  [DIAGNOSIS] Accelergy ran, but the generated ERT is empty or invalid.")
            return None

        # 健壮性检查 2: 我们只定义了一个组件，所以应该只有一个 energy table
        if len(ert_data['ERT']['tables']) != 1:
            print(f"\n  [DIAGNOSIS] Expected 1 table in ERT, but found {len(ert_data['ERT']['tables'])}.")
            return None
        
        target_table = ert_data['ERT']['tables'][0]
        
        # 健壮性检查 3: table 中是否有 'actions' 列表？
        if 'actions' not in target_table or not target_table['actions']:
             print(f"\n  [DIAGNOSIS] Table '{target_table.get('name')}' found, but it has no 'actions' defined.")
             return None

        # 我们只请求了 'read'，所以它应该是第一个 action
        energy = target_table['actions'][0]['energy']
        return energy
        
    except (IOError, KeyError, IndexError, TypeError) as e:
        print(f"\n  [DIAGNOSIS] Failed to parse ERT file with an unexpected error: {e}")
        print(f"  Please check the contents of {ert_path}")
        return None

def fit_energy_model(sizes_kb, energies_pj):
    """使用线性回归拟合能量模型系数。"""
    X = np.array(sizes_kb).reshape(-1, 1)
    y = np.array(energies_pj)

    model = LinearRegression()
    model.fit(X, y)
    
    capacity_coeff = model.coef_[0]
    base_epa = model.intercept_

    return base_epa, capacity_coeff


def main():
    """主校准流程。"""
    print("Starting Accelergy calibration process for SRAM...")

    if os.path.exists(WORKSPACE_DIR):
        shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR)

    collected_data = []

    for size in SRAM_SIZES_KB:
        print(f"\nProcessing SRAM size: {size} KB")
        
        print(f"  - Generating Accelergy YAML files...")
        generate_accelergy_inputs(size)
        
        if not run_accelergy_simulation():
            print("  - Skipping this data point due to Accelergy failure.")
            continue
            
        print(f"  - Parsing ERT.yaml for energy values...")
        energy = parse_accelergy_output()
        
        if energy is not None:
            print(f"  - Success! Collected energy: {energy:.4f} pJ")
            collected_data.append({'size': size, 'energy': energy})

    if len(collected_data) < 2:
        print("\nCalibration failed: Not enough data points collected to perform fitting.")
        print("Please check the diagnosis messages above.")
        return

    sizes = [d['size'] for d in collected_data]
    energies = [d['energy'] for d in collected_data]

    print("\n" + "="*80)
    print("PERFORMING LINEAR REGRESSION ON COLLECTED SRAM ENERGY DATA")
    print("="*80)
    print(f"Data points collected: {len(sizes)}")
    for item in collected_data:
        print(f"  - Size: {item['size']:>4} KB, Energy: {item['energy']:.4f} pJ")
    
    base_epa, capacity_coeff = fit_energy_model(sizes, energies)

    print("\n--- FIT RESULTS ---")
    print(f"Model: Energy(pJ) = BASE_EPA + CAPACITY_COEFF * Size(KB)")
    print(f"  -> Intercept (BASE_EPA): {base_epa:.6f}")
    print(f"  -> Slope (CAPACITY_COEFF): {capacity_coeff:.6f}")

    print("\n" + "="*80)
    print("ACTION REQUIRED: PLEASE UPDATE 'dosa/config.py'")
    print("="*80)
    print("Copy and paste the following code block into your 'dosa/config.py' file,\n"
          "replacing the existing SRAM energy parameters:\n")

    dram_typical_energy_pj = 50.0  # 这是一个假设值, 真实值需要单独仿真
    
    print("# --- 2. 存储组件 (基于Accelergy仿真拟合) ---")
    print(f"# L1/L2 SRAM (Fitted from Accelergy simulations at {TECHNOLOGY})")
    print(f"self.SRAM_BASE_EPA_PJ = {base_epa:.6f}")
    print(f"self.SRAM_CAPACITY_COEFF_PJ_PER_KB = {capacity_coeff:.6f}\n")
    
    print(f"# L3 DRAM (Based on a typical value, placeholder for real simulation)")
    print(f"self.L3_DRAM_EPA_PJ = {dram_typical_energy_pj:.6f}")
    print("\nCalibration complete.")


if __name__ == "__main__":
    main()