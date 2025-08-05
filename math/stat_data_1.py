import json
import numpy as np
import os
from collections import defaultdict

def analyze_by_discovered_groups(base_dir="."):
    """
    自动发现在所有 dialogue_gpt-4.1_test* 子目录中的所有 
    dialogue_*_result.json 文件，然后按文件名分组进行统计分析。
    """
    
    # 使用 defaultdict 可以简化代码，当一个新文件名出现时，自动为其创建空列表。
    # 结构: {'filename': {'values': [], 'sources': []}}
    data_groups = defaultdict(lambda: {'values': [], 'sources': []})

    print("--- Phase 1: Discovering files and gathering data ---")
    
    # 步骤1：遍历所有目录和文件，发现并收集数据
    for i in range(1, 9): # 遍历 dialogue_gpt-4.1_test1 到 8
        dir_name = f"dialogue_gpt-4.1_test{i}"
        full_dir_path = os.path.join(base_dir, dir_name)

        if not os.path.isdir(full_dir_path):
            continue

        for filename in os.listdir(full_dir_path):
            # 检查文件名是否符合 'dialogue_..._result.json' 格式
            if filename.startswith("dialogue_") and filename.endswith("_result.json"):
                filepath = os.path.join(full_dir_path, filename)
                relative_path = os.path.join(dir_name, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "rounds_used" in data:
                            value = data["rounds_used"]
                            # 将数据添加到对应的文件名分组中
                            data_groups[filename]['values'].append(value)
                            data_groups[filename]['sources'].append(relative_path)
                        else:
                             print(f"  ⚠️  Warning: 'rounds_used' key not found in {relative_path}")
                except Exception as e:
                    print(f"  ❌ Error processing file {relative_path}: {e}")

    if not data_groups:
        print("\nError: No valid data files found matching the pattern 'dialogue_*_result.json'. Analysis aborted.")
        return

    print(f"\n--- Phase 2: Analyzing {len(data_groups)} discovered file groups ---")

    # 步骤2：遍历收集到的数据组，逐个进行分析和报告
    for filename, data in sorted(data_groups.items()):
        
        values = data['values']
        sources = data['sources']

        print(f"\n========================================================")
        print(f"📊 Analysis for file group: {filename}")
        print(f"========================================================")

        if len(values) < 2:
            print(f"Insufficient data for '{filename}' (found {len(values)} instance(s)). Meaningful statistics require at least 2 data points.")
            continue

        values_array = np.array(values)
        mean_value = np.mean(values_array)
        variance_value = np.var(values_array)
        std_dev_value = np.std(values_array)

        print(f"--- Statistics for [{filename}] ---")
        print(f"Data points found: {len(values)}")
        print(f"Values: {values}")
        print(f"Mean (μ): {mean_value:.4f}")
        print(f"Variance (σ²): {variance_value:.4f}")
        print(f"Standard Deviation (σ): {std_dev_value:.4f}")
        print("---------------------------------------------")

        # 在当前组内筛选标记数据
        lower_bound = mean_value - 2 * std_dev_value
        upper_bound = mean_value + 2 * std_dev_value

        print(f"\n--- Outlier Report for [{filename}] ---")
        print(f"Detection bounds: Value < {lower_bound:.4f} or > {upper_bound:.4f}")
        
        marked_data_found = False
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                marked_data_found = True
                diff_from_mean = value - mean_value
                sign = "+" if diff_from_mean > 0 else ""
                print(f"  📍 Marked: {value:<5} | Source: {sources[i]:<40} | (Deviation from group mean: {sign}{diff_from_mean:.2f})")

        if not marked_data_found:
            print("  No outliers detected in this group.")
            
    print(f"\n========================================================")
    print("✅ All discovered file groups have been analyzed.")
    print(f"========================================================")


if __name__ == "__main__":
    analyze_by_discovered_groups()