import re
from pathlib import Path

def parse_debug_file(debug_file):
    """
    解析 debug.txt，提取 traffic 信息
    返回 dict: { "L2_Scratchpad_to_L0_Registers": {"type":"Fill", "bytes":73728.0}, ... }
    """
    traffic_pattern = re.compile(r"\[DEBUG\]\s+(\w+)\s+traffic:\s+(\S+)\s+->\s+([\d\.]+)\s+bytes")
    results = {}
    with open(debug_file, "r") as f:
        for line in f:
            m = traffic_pattern.search(line)
            if m:
                t_type, path, value = m.groups()
                results[path] = {"type": t_type, "bytes": float(value)}
    return results


def parse_timeloop_stats(stats_file):
    """
    解析 timeloop-mapper.stats.txt
    返回 dict，按 Level->Name->Metrics
    """
    results = {}
    current_level = None
    current_name = None
    with open(stats_file, "r") as f:
        for line in f:
            line = line.strip()
            # Level 标识
            if line.startswith("Level "):
                current_level = line
            # 模块名
            elif line.startswith("===") and line.endswith("==="):
                current_name = line.strip("= ").strip()
                results[current_name] = {}
            elif ":" in line and current_name:
                key, val = [x.strip() for x in line.split(":", 1)]
                # 提取数字
                try:
                    val_num = float(val.split()[0])
                    results[current_name][key] = val_num
                except ValueError:
                    results[current_name][key] = val
    return results


def compare_results(debug_data, timeloop_data):
    """
    做简单的差分比较：
    - 打印 Debug traffic vs Timeloop STATS
    """
    print("\n=== 差分测试结果 ===\n")
    for path, info in debug_data.items():
        dbg_val = info["bytes"]
        dbg_type = info["type"]

        # 在 timeloop_data 里找对应的模块
        matched = None
        for name, stats in timeloop_data.items():
            if "Scalar reads (per-instance)" in stats and "Read" in dbg_type:
                matched = name
            if "Scalar fills (per-instance)" in stats and "Fill" in dbg_type:
                matched = name
            if "Scalar updates (per-instance)" in stats and "Writeback" in dbg_type:
                matched = name

        if matched:
            print(f"{dbg_type} {path}: Debug={dbg_val:.2f} bytes | Timeloop({matched}) scalar accesses ≈ {timeloop_data[matched].get('Scalar reads (per-instance)', timeloop_data[matched].get('Scalar fills (per-instance)', timeloop_data[matched].get('Scalar updates (per-instance)', 0.0)))}")
        else:
            print(f"{dbg_type} {path}: Debug={dbg_val:.2f} bytes | Timeloop=未找到对应字段")


if __name__ == "__main__":
    # 修改成你实际的文件路径
    debug_file = Path("debug.txt")
    stats_file = Path("output/validation_workspace_1/timeloop-mapper.stats.txt")

    debug_data = parse_debug_file(debug_file)
    timeloop_data = parse_timeloop_stats(stats_file)
    compare_results(debug_data, timeloop_data)

