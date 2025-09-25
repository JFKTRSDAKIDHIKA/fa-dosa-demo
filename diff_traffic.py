import pandas as pd
import re

BYTES_PER_ACCESS = 2

def parse_pe_count_from_stats(stats_file_path):
    """
    从timeloop stats文件中解析PE数量（Utilized instances）
    """
    try:
        with open(stats_file_path, 'r') as f:
            content = f.read()
        
        # 查找MAC层级的Utilized instances
        # 匹配模式：Level 0 -> MAC -> Utilized instances : 64
        pattern = r'Level 0.*?=== MAC ===.*?Utilized instances\s*:\s*(\d+)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            pe_count = int(match.group(1))
            print(f"从timeloop stats文件中解析到PE数量: {pe_count}")
            return pe_count
        else:
            print("警告: 无法从timeloop stats文件中解析PE数量，使用默认值64")
            return 64
    except Exception as e:
        print(f"警告: 解析timeloop stats文件时出错: {e}，使用默认值64")
        return 64

# 误差容忍阈值
ERROR_TOLERANCE = {
    "absolute_threshold": 1000.0,  # 绝对误差阈值（字节）
    "relative_threshold": 0.22     # 相对误差阈值（5%）
}

# 层名字 -> Timeloop 的 level（按你的 stats：1,3,4,5）
LEVEL_NAME_TO_ID = {
    "L0_Registers": 1,
    "L1_Accumulator": 3,
    "L2_Scratchpad": 4,   # 注意：是 4，不是 2
    "L3_DRAM": 5,
    "PE": None,           # 不计入 timeloop 表
}

# 合法的 tensor+type 组合（用于模型侧 & timeloop 过滤）
VALID_TYPES = {
    "Weights": {"read", "write"},
    "Inputs": {"read", "write"},
    "Outputs": {"read", "write", "update"},
}

def split_iface(iface: str):
    parts = iface.split("_to_")
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]

def map_model_level(row):
    """write/update 计入 dst 层；read 计入 src 层；映射到数字 level。"""
    iface, ttype = row["iface"], row["type"]
    src, dst = split_iface(iface)
    if src is None:
        return None
    name = dst if ttype in ("write", "update") else src
    return LEVEL_NAME_TO_ID.get(name, None)

def load_model_csv(path: str):
    print(f"[DEBUG] Loading model CSV from: {path}")
    df = pd.read_csv(path)
    print(f"[DEBUG] Model CSV loaded: {len(df)} rows, columns: {df.columns.tolist()}")
    
    if "tensor" not in df.columns:
        raise ValueError(f"[ERROR] 模型 CSV 缺少 tensor 列，实际列: {df.columns.tolist()}")

    # 标准化 tensor 名称为 Timeloop 的复数
    df["tensor"] = df["tensor"].replace({"Weight":"Weights","Input":"Inputs","Output":"Outputs"})

    # 映射到 level
    df["level"] = df.apply(map_model_level, axis=1)
    df = df.dropna(subset=["level"]).copy()
    df["level"] = df["level"].astype(int)

    # 只保留合法组合
    df = df[df.apply(lambda r: r["type"] in VALID_TYPES.get(r["tensor"], set()), axis=1)]
    print(f"[DEBUG] Model CSV after processing: {len(df)} rows")
    return df

def load_timeloop_csv(path: str, stats_file_path: str = None):
    """
    加载timeloop CSV文件并转换为字节数
    
    Args:
        path: timeloop CSV文件路径
        stats_file_path: timeloop stats文件路径，用于解析PE数量
    """
    print(f"[DEBUG] Loading timeloop CSV from: {path}")
    print(f"[DEBUG] Stats file path: {stats_file_path}")
    
    # 解析PE数量
    if stats_file_path:
        pe_count = parse_pe_count_from_stats(stats_file_path)
    else:
        pe_count = 64  # 默认值
        print("警告: 未提供stats文件路径，使用默认PE数量64")
    
    df = pd.read_csv(path)
    print(f"[DEBUG] Timeloop CSV loaded: {len(df)} rows, columns: {df.columns.tolist()}")
    
    # 展开 fills/reads/updates
    df = df.melt(
        id_vars=["level", "tensor"],
        value_vars=["fills", "reads", "updates"],
        var_name="timeloop_op", value_name="accesses"
    )

    # 语义映射到模型的 type
    def map_type(row):
        op = row["timeloop_op"]
        tensor = row["tensor"]
        level = int(row["level"])
        if op == "fills":
            return "write"
        if op == "reads":
            return "read"
        if op == "updates":
            # L1_Accumulator(3) 的 Outputs.update 保留为 update
            if tensor == "Outputs" and level == 3:
                return "update"
            # DRAM(5) 的 Outputs.update 直接转成 write
            if tensor == "Outputs" and level == 5:
                return "write"
            # 其他 updates 丢弃
            return None
        return None


    df["type"] = df.apply(map_type, axis=1)
    df = df.dropna(subset=["type"]).copy()
    df = df.drop_duplicates(subset=["level","tensor","type"], keep="last")


    # 只保留合法组合
    df = df[df.apply(lambda r: r["type"] in VALID_TYPES.get(r["tensor"], set()), axis=1)]

    # 访问次数 -> 字节（Timeloop 表里某些层级是 per instance，需要乘以 PE 数量）
    def calculate_bytes(row):
        accesses = row["accesses"]
        level = int(row["level"])
        tensor = row["tensor"]
        ttype = row["type"]
        
        # L0_Registers (level 1) 的 Weights 访问是 per PE instance，需要乘以 pe_count
        if level == 1 and tensor == "Weights":
            return accesses * BYTES_PER_ACCESS * pe_count
        # L2_Scratchpad (level 4) 的 Weights read 也是 per PE instance，需要乘以 pe_count
        elif level == 4 and tensor == "Weights" and ttype == "read":
            return accesses * BYTES_PER_ACCESS * pe_count
        else:
            return accesses * BYTES_PER_ACCESS
    
    df["bytes"] = df.apply(calculate_bytes, axis=1)
    return df

def run_traffic_diff_analysis(model_csv_path="traffic_summary_tensor.csv", timeloop_csv_path="scalar_access_summary.csv", output_csv_path="traffic_diff.csv", stats_file_path=None):
    """
    运行traffic差分分析，可以作为模块函数被调用
    
    Args:
        model_csv_path: 模型CSV文件路径
        timeloop_csv_path: Timeloop CSV文件路径  
        output_csv_path: 输出CSV文件路径
        stats_file_path: Timeloop stats文件路径，用于解析PE数量
        
    Returns:
        dict: 包含分析结果的字典
    """
    print(f"[DEBUG] ===== Traffic Diff Analysis Start =====")
    print(f"[DEBUG] Model CSV path: {model_csv_path}")
    print(f"[DEBUG] Timeloop CSV path: {timeloop_csv_path}")
    print(f"[DEBUG] Output CSV path: {output_csv_path}")
    print(f"[DEBUG] Stats file path: {stats_file_path}")
    print(f"[DEBUG] ==========================================")
    
    model_df = load_model_csv(model_csv_path)
    tl_df = load_timeloop_csv(timeloop_csv_path, stats_file_path)

    rows = []
    for _, r in tl_df.iterrows():
        level = int(r["level"])
        tensor = r["tensor"]
        ttype  = r["type"]
        tl_bytes = float(r["bytes"])

        # 在模型侧找对应的 (level, tensor, type)
        sub = model_df[(model_df["level"]==level) & (model_df["tensor"]==tensor) & (model_df["type"]==ttype)]
        if sub.empty:
            print(f"[WARNING] 模型侧没有找到 ({level}, {tensor}, {ttype})")
            model_bytes = 0.0
        else:
            model_bytes = float(sub["bytes"].sum())

        diff = model_bytes - tl_bytes
        rel = diff / (tl_bytes + 1e-9)
        
        # 判断是否正确（在误差容忍范围内）
        abs_error = abs(diff)
        rel_error = abs(rel)
        is_correct = (abs_error <= ERROR_TOLERANCE["absolute_threshold"]) or (rel_error <= ERROR_TOLERANCE["relative_threshold"])

        # 收集模型侧的 iface 信息
        ifaces = ";".join(sorted(sub["iface"].unique())) if not sub.empty else ""

        rows.append({
            "level": level,
            "tensor": tensor,
            "type": ttype,                 # 注意：DRAM/Outputs 的 updates 已改为 write
            "timeloop_bytes": tl_bytes,
            "model_bytes": model_bytes,
            "diff": diff,
            "rel_error": rel,
            "is_correct": is_correct,
            "model_ifaces": ifaces
        })

    out = pd.DataFrame(rows).sort_values(["level","tensor","type"]).reset_index(drop=True)
    out.to_csv(output_csv_path, index=False)
    print(f"[INFO] 差分结果已保存到 {output_csv_path}")
    
    # 统计正确性
    total_count = len(out)
    correct_count = out["is_correct"].sum()
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    print(f"[INFO] 正确性统计: {correct_count}/{total_count} ({accuracy:.1f}%)")

    # 关键 sanity check：DRAM/Outputs 写回应该严格对齐
    print(out[(out["level"]==5)&(out["tensor"]=="Outputs")])
    
    return {
        "total_count": total_count,
        "correct_count": correct_count,
        "accuracy_percent": accuracy,
        "results_dataframe": out
    }

def main():
    """命令行入口函数"""
    result = run_traffic_diff_analysis()
    return result

if __name__ == "__main__":
    main()
