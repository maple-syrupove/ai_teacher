import re
import json
from pathlib import Path
import shutil

ERROR_DIR = "error_math_new"
BACKUP_DIR = "error_math_new_backup"

# 备份原始文件
Path(BACKUP_DIR).mkdir(exist_ok=True)
for f in Path(ERROR_DIR).glob("*.json"):
    shutil.copy(f, Path(BACKUP_DIR) / f.name)

# 修复非法反斜杠的函数
def fix_illegal_backslash(text):
    # 只修复字符串中的非法\，不过度破坏合法转义
    # 先修复 "...\..." 里不是合法转义的反斜杠
    def repl(m):
        s = m.group(0)
        # 合法转义: \\, \", \/, \b, \f, \n, \r, \t, \u
        s = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', s)
        return s
    return re.sub(r'"(.*?)(?<!\\)"', repl, text)

# 批量修复
total, fixed, failed = 0, 0, 0
for f in Path(ERROR_DIR).glob("*.json"):
    total += 1
    with open(f, "r", encoding="utf-8") as file:
        raw = file.read()
    fixed_text = fix_illegal_backslash(raw)
    try:
        # 检查修复后能否正常解析
        json.loads(fixed_text)
        with open(f, "w", encoding="utf-8") as file:
            file.write(fixed_text)
        fixed += 1
    except Exception as e:
        print(f"修复失败: {f} -> {e}")
        failed += 1
print(f"共处理{total}个文件，修复成功{fixed}个，失败{failed}个。备份已存于{BACKUP_DIR}") 