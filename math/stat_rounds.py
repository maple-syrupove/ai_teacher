import os
import json
import numpy as np
from glob import glob

def stat_student_dir(student_dir):
    files = glob(os.path.join(student_dir, '*.json'))
    rounds = []
    tokens = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fin:
                data = json.load(fin)
            if 'rounds_used' in data and 'teacher_token_avg' in data:
                rounds.append(data['rounds_used'])
                tokens.append(data['teacher_token_avg'])
        except Exception as e:
            print(f"Error reading {f}: {e}")
    if rounds and tokens:
        return {
            'count': len(rounds),
            'rounds_mean': float(np.mean(rounds)),
            'rounds_std': float(np.std(rounds, ddof=1)) if len(rounds) > 1 else 0.0,
            'tokens_mean': float(np.mean(tokens)),
            'tokens_std': float(np.std(tokens, ddof=1)) if len(tokens) > 1 else 0.0
        }
    else:
        return None

def stat_all(base_dir):
    result = {}
    for i in range(6):
        sub = os.path.join(base_dir, f'student{i}', 'student0')
        if os.path.isdir(sub):
            stat = stat_student_dir(sub)
            if stat:
                result[f'student{i}'] = stat
    return result

def write_stat_txt(base_dir, stat):
    out_path = os.path.join(base_dir, 'statistics.txt')
    with open(out_path, 'w', encoding='utf-8') as fout:
        for k, v in stat.items():
            fout.write(f'{k}:\n')
            fout.write(f'  count: {v["count"]}\n')
            fout.write(f'  rounds_used mean: {v["rounds_mean"]:.2f}, std: {v["rounds_std"]:.2f}\n')
            fout.write(f'  teacher_token_avg mean: {v["tokens_mean"]:.2f}, std: {v["tokens_std"]:.2f}\n')
            fout.write('\n')
    print(f'统计结果已写入: {out_path}')

if __name__ == '__main__':
    dirs = [
        'dialogue_20250721_155441_r1'
    ]
    for d in dirs:
        stat = stat_all(d)
        write_stat_txt(d, stat) 