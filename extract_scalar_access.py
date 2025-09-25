import re
import pandas as pd

def parse_scalar_access(file_path):
    results = []
    current_level = None
    current_tensor = None

    # 匹配正则
    level_re = re.compile(r"^Level\s+(\d+)")
    tensor_re = re.compile(r"^(Inputs|Outputs|Weights):")
    scalar_re = re.compile(r"Scalar (reads|fills|updates).*?:\s+(\d+)")

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # 判断层级
            m_level = level_re.match(line)
            if m_level:
                current_level = int(m_level.group(1))
                continue

            # 判断张量
            m_tensor = tensor_re.match(line)
            if m_tensor:
                current_tensor = m_tensor.group(1)
                continue

            # 匹配 scalar 读写更新
            m_scalar = scalar_re.match(line)
            if m_scalar and current_level is not None and current_tensor is not None:
                stype, val = m_scalar.group(1), int(m_scalar.group(2))
                results.append({
                    "level": current_level,
                    "tensor": current_tensor,
                    "type": stype,   # reads / fills / updates
                    "count": val
                })

    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "/root/fa-dosa-demo/output/validation_workspace_1/timeloop-mapper.stats.txt"  # 默认文件路径
    
    data = parse_scalar_access(file_path)

    # 转成 DataFrame
    df = pd.DataFrame(data)
    df_pivot = df.pivot_table(
        index=["level", "tensor"],
        columns="type",
        values="count",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    print(df_pivot)

    # 如果要存成 CSV
    df_pivot.to_csv("scalar_access_summary.csv", index=False)
