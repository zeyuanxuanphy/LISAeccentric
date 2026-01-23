# 文件路径: LISAeccentric/__init__.py

# 1. 从同级目录的 core.py 中导入核心类
try:
    # 这里的 LISAeccentric 是类名，CompactBinary 是数据类
    from .core import LISAeccentric as _CoreEngine
    from .core import CompactBinary
except ImportError as e:
    # 错误提示：帮助调试路径问题
    raise ImportError(f"LISAeccentric package initialization failed. Could not import 'core.py'.\nDetails: {e}")

# 2. 【自动实例化】
# 在包被导入时，创建一个全局的单例对象。
# 这样外界不需要 tool = LISAeccentric()，直接用包名即可调用。
_default_instance = _CoreEngine()

# 3. 【挂载功能模块】
# 将实例中的子模块赋值给包的顶层变量。
GN = _default_instance.GN
GC = _default_instance.GC
Field = _default_instance.Field
Waveform = _default_instance.Waveform
Noise = _default_instance.Noise  # <--- 【关键修复】：这里加上了 Noise 模块

# 4. 【暴露数据类】
# 允许用户直接使用 LISAeccentric.CompactBinary(...)
CompactBinary = CompactBinary

# 5. 定义包的公共接口
# 这里决定了 from LISAeccentric import * 会导入什么
__all__ = [
    'GN',
    'GC',
    'Field',
    'Waveform',
    'Noise',          # <--- 【关键修复】：这里也要加上
    'CompactBinary'
]

# print("LISAeccentric package initialized successfully.")