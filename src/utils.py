from functools import wraps
import time
from pathlib import Path


# 项目根路径
ROOT_DIR = Path.cwd()
if ROOT_DIR.joinpath('pinyin.py').exists(): # 如果当前路径是 'src'，那么回到项目根路径
    ROOT_DIR = ROOT_DIR.parent


def timer(fn):
    """
    装饰器，用于函数运行计时。
    """

    @wraps(fn)
    def wrapper(*args, **kw):
        print(f'Start executing {fn.__name__}.')
        start_time = time.time()
        result = fn(*args, **kw)
        end_time = time.time()
        print(f'{fn.__name__} executed in {end_time - start_time}s')
        return result
    return wrapper
