import shutil, os, sys
import hashlib
from A25.global_config import GlobalConfig as cfg
from A25.UTIL.colorful import *

def calculate_file_hash(file_path):
    """计算文件的MD5哈希值"""
    if not os.path.exists(file_path):
        return None
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_cache_path(wl_vid, ir_vid, operation):
    """根据输入文件的哈希值生成缓存文件路径"""
    wl_hash = calculate_file_hash(wl_vid)
    ir_hash = calculate_file_hash(ir_vid)
    if not wl_hash or not ir_hash:
        return None
    
    cache_dir = os.path.join(cfg.outputdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_filename = f"{operation}_{wl_hash[:8]}_{ir_hash[:8]}"
    return os.path.join(cache_dir, cache_filename)

def with_dual_cache(func):
    """缓存装饰器"""
    def wrapper(*args, **kwargs):
        operation = func.__name__
        if len(args) >= 2:  # 确保有两个输入视频参数
            wl_vid, ir_vid = args[0], args[1]
            cache_path = get_cache_path(wl_vid, ir_vid, operation)
            
            if cache_path:
                cache_dir = os.path.dirname(cache_path)
                cache_file_ = os.listdir(cache_dir)
                cache_file = []
                for f in cache_file_:
                    if f.startswith(os.path.basename(cache_path)):
                        cache_file.append(os.path.join(cache_dir, f))
                
                if len(cache_file) > 0:
                    print黄(f"使用缓存: {cache_file}")
                    return tuple(iter(cache_file))
                
        result = func(*args, **kwargs)
        
        # 如果结果是元组（比如wl_vid和ir_vid），保存两个文件
        if isinstance(result, tuple):
            cache_path = get_cache_path(wl_vid, ir_vid, operation)
            if cache_path:
                for i, res in enumerate(result):
                    if not os.path.exists(res):
                        return result

                for i, res in enumerate(result):
                    res_cache_path = cache_path + '-' + str(i)
                    shutil.copy2(res, res_cache_path)
                return result
        else:
            cache_path = get_cache_path(wl_vid, ir_vid, operation)
            if cache_path and os.path.exists(result):
                shutil.copy2(result, cache_path)
                return cache_path
                
        return result
    return wrapper