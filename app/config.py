"""
config.py - 配置导入适配层
将 'from config import get_config' 转发到正确的配置模块
"""
import os
import sys


# ==================== 默认配置（永不失败的最后保障） ====================

_DEFAULTS = {
    "app": {"version": "1.0.0", "environment": "development"},
    "models": {"cache_enabled": True, "ensemble_enabled": True, "cache_ttl": 3600, "max_cache_size": 1000},
    "features": {"cache_enabled": True, "parallel_enabled": True, "cache_size": 1000},
}


def _deep_merge(base, overrides):
    """深度合并两个字典"""
    merged = dict(base)
    for k, v in overrides.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


# ==================== 配置代理类 ====================

class _ConfigProxy:
    """轻量级配置代理，支持 dict 风格的 .get() 访问"""

    def __init__(self, cfg_dict):
        self._cfg = cfg_dict

    def get(self, key, default=None):
        """支持点分隔键：'app.version'、'models.cache_enabled'"""
        keys = key.split('.')
        val = self._cfg
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, key):
        return self.get(key) is not None

    def to_dict(self):
        return self._cfg


# ==================== 全局实例 ====================

_instance = None


def get_config():
    """
    返回一个兼容的 _ConfigProxy 对象。
    优先使用项目根目录的 config 包，其次使用 config_loader.py，
    最后永不失败地返回默认配置。
    """
    global _instance

    if _instance is not None:
        return _instance

    cfg_dict = {}

    # 方法1：使用项目根目录的 config 包（config/__init__.py）
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_init = os.path.join(root, 'config', '__init__.py')
        if os.path.exists(config_init):
            import importlib.util
            spec = importlib.util.spec_from_file_location("_config_root", config_init)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_config_root"] = mod
            spec.loader.exec_module(mod)
            real = mod.config
            if hasattr(real, 'to_dict'):
                cfg_dict = real.to_dict()
            elif isinstance(real, dict):
                cfg_dict = real
            _instance = _ConfigProxy(_deep_merge(_DEFAULTS, cfg_dict))
            return _instance
    except Exception:
        pass

    # 方法2：使用 config_loader.py
    try:
        from .config_loader import get_config_loader
        loader = get_config_loader()
        if hasattr(loader, 'load_config'):
            full_cfg = loader.load_config()
            if hasattr(full_cfg, 'to_dict'):
                cfg_dict = full_cfg.to_dict()
            elif hasattr(full_cfg, '__dict__'):
                cfg_dict = {k: v for k, v in full_cfg.__dict__.items() if not k.startswith('_')}
        _instance = _ConfigProxy(_deep_merge(_DEFAULTS, cfg_dict))
        return _instance
    except Exception:
        pass

    # 方法3：永不失败，返回默认值
    _instance = _ConfigProxy(_DEFAULTS)
    return _instance


# 模块级默认实例（供 'from config import config' 使用）
config = get_config()
