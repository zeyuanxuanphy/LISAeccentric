# 文件路径: LISAeccentric/__init__.py

# 1. 从同级目录的 core.py 中导入核心类 以及 控制函数
try:
    # 导入核心类
    from .core import LISAeccentric as _CoreEngine
    from .core import CompactBinary

    # 【关键修改】：显式导入控制函数
    from .core import set_output_control, set_verbose

except ImportError as e:
    # 错误提示：帮助调试路径问题
    raise ImportError(f"LISAeccentric package initialization failed. Could not import 'core.py'.\nDetails: {e}")

# 2. 【自动实例化】
_default_instance = _CoreEngine()

# 3. 【挂载功能模块】
GN = _default_instance.GN
GC = _default_instance.GC
Field = _default_instance.Field
Waveform = _default_instance.Waveform
Noise = _default_instance.Noise

# 4. 【暴露数据类】
CompactBinary = CompactBinary

# 5. 定义包的公共接口
# 这里决定了 from LISAeccentric import * 会导入什么
__all__ = [
    'GN',
    'GC',
    'Field',
    'Waveform',
    'Noise',
    'CompactBinary',
    'set_output_control',  # <--- 【关键修改】：加入导出列表
    'set_verbose'  # <--- 【关键修改】：加入导出列表
]

# print("LISAeccentric package initialized successfully.")