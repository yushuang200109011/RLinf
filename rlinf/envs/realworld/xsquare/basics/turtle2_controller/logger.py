import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

class Logger:
    def __init__(self, name, log_dir="/home/arm/logs/turtle2_controller", level=logging.DEBUG):
        """初始化日志记录器

        Args:
            name: logger名称(通常使用__name__)
            log_dir: 日志目录路径
            level: 日志级别
        """
        # 创建日志目录
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # 配置日志格式
        self._formatter = logging.Formatter(
            fmt='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s][%(threadName)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 创建logger实例
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        self._log_counters = {}

        # 如果已经配置过handler则不再重复添加
        if not self.logger.handlers:
            self._setup_handlers(log_dir, name)

    def _setup_handlers(self, log_dir, name):
        """配置日志处理器"""
        # 时间轮转文件处理器(每小时)
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, f'{name}.logging'),
            when='H', # 每1小时拆分一个日志文件
            interval=1,
            backupCount=24*7, # 最多保留7天的全量日志
            encoding='utf-8'
        )
        file_handler.suffix = "%Y%m%d_%H.log"  # 非活跃文件后缀格式
        file_handler.setFormatter(self._formatter)

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs, stacklevel=2)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs, stacklevel=2)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs, stacklevel=2)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs, stacklevel=2)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs, stacklevel=2)

    def debug_every_n(self, n, msg, *args, **kwargs):
        self._log_every_n("debug", n, msg, *args, **kwargs)

    def info_every_n(self, n, msg, *args, **kwargs):
        self._log_every_n("info", n, msg, *args, **kwargs)

    def warning_every_n(self, n, msg, *args, **kwargs):
        self._log_every_n("warning", n, msg, *args, **kwargs)

    def error_every_n(self, n, msg, *args, **kwargs):
        self._log_every_n("error", n, msg, *args, **kwargs)

    def critical_every_n(self, n, msg, *args, **kwargs):
        self._log_every_n("critical", n, msg, *args, **kwargs)

    def _log_every_n(self, level, n, msg, *args, **kwargs):
        """内部方法:实现每N次记录"""
        key = f"{level}:{msg}"  # 使用日志级别和消息作为唯一键
        self._log_counters[key] = self._log_counters.get(key, 0) + 1

        if self._log_counters[key] % n == 1:  # 第一次或每N次记录
            getattr(self.logger, level)(msg, *args, **kwargs, stacklevel=3)


# 全局日志实例
logger = Logger("robocontrol", level=logging.INFO)
