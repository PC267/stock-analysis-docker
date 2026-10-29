import os
from pathlib import Path
import platform
from typing import List


class ReportDirectory:
    """报告目录管理类"""

    @staticmethod
    def _get_desktop_path() -> Path:
        """获取适合存放报告的根目录（桌面或公共存储）"""
        system = platform.system()

        if system == 'Windows':
            # Windows：使用注册表获取真实桌面路径
            import winreg
            try:
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                     r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders')
                desktop = winreg.QueryValueEx(key, 'Desktop')[0]
                winreg.CloseKey(key)
                return Path(desktop)
            except Exception:
                # 备用方案：用户目录下的 Desktop
                return Path.home() / 'Desktop'

        elif system == 'Linux' and 'ANDROID_ROOT' in os.environ:
            # 安卓环境：使用外部存储（用户可访问的共享存储）
            # 优先使用环境变量 EXTERNAL_STORAGE，通常为 /storage/emulated/0
            external = os.environ.get('EXTERNAL_STORAGE', '/storage/emulated/0')
            desktop_candidate = Path(external) / 'Desktop'
            # 如果 Desktop 文件夹不存在，也可以直接返回外部存储根目录
            # 这里仍然返回 Desktop 子目录，但代码调用方会负责创建
            return desktop_candidate
        else:
            # macOS / 普通 Linux 桌面：检查常见桌面目录
            home = Path.home()
            for candidate in (home / 'Desktop', home / '桌面'):
                if candidate.exists():
                    return candidate
            return home / 'Desktop'

    # 基础目录路径：容器内统一报告输出目录
    BASE_DIR = Path(os.environ.get("REPORT_OUTPUT_DIR", "/work/app/results/reports"))

    @classmethod
    def get_base_path(cls) -> Path:
        """获取基础目录路径"""
        return cls.BASE_DIR

    @classmethod
    def ensure_directory_exists(cls) -> bool:
        """确保目录存在，不存在则创建"""
        try:
            cls.BASE_DIR.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"创建目录失败: {e}")
            return False

    @classmethod
    def get_subdirectory(cls, subdir: str) -> Path:
        """获取子目录路径"""
        return cls.BASE_DIR / subdir

    @classmethod
    def get_file_path(cls, filename: str) -> Path:
        """获取文件完整路径"""
        return cls.BASE_DIR / filename

    @classmethod
    def list_files(cls, pattern: str = "*") -> List[Path]:
        """列出目录中的文件"""
        if cls.BASE_DIR.exists():
            return list(cls.BASE_DIR.glob(pattern))
        return []

    @classmethod
    def is_directory_empty(cls) -> bool:
        """检查目录是否为空"""
        if not cls.BASE_DIR.exists():
            return True
        return not any(cls.BASE_DIR.iterdir())


# 简便函数
def get_report_dir() -> str:
    """获取报告目录字符串"""
    return str(ReportDirectory.BASE_DIR)


def get_report_path() -> Path:
    """获取报告目录Path对象"""
    return ReportDirectory.BASE_DIR


def join_path(*args) -> Path:
    """拼接路径"""
    return ReportDirectory.BASE_DIR.joinpath(*args)


# 常用快捷方式
REPORT_DIR = str(ReportDirectory.BASE_DIR)
REPORT_PATH = ReportDirectory.BASE_DIR

if __name__ == "__main__":
    print(f"报告目录: {get_report_dir()}")
    if ReportDirectory.ensure_directory_exists():
        print("目录创建/验证成功")
        # 额外打印真实绝对路径，帮助确认位置
        print(f"实际路径: {ReportDirectory.BASE_DIR.resolve()}")
    else:
        print("目录创建/验证失败")