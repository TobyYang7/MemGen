import os
import logging

from larm.common import dist_utils 


class ColoredFormatter(logging.Formatter):
    """自定义日志格式化器，为不同级别的日志添加颜色"""
    
    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[0m',        # 默认颜色（白色）
        'WARNING': '\033[91m',    # 红色
        'ERROR': '\033[91m',      # 红色
        'CRITICAL': '\033[91m\033[1m',  # 红色加粗
    }
    RESET = '\033[0m'  # 重置颜色
    
    def format(self, record):
        # 获取日志级别对应的颜色
        log_color = self.COLORS.get(record.levelname, self.RESET)
        
        # 格式化日志消息
        formatted = super().format(record)
        
        # 为整行日志添加颜色
        return f"{log_color}{formatted}{self.RESET}"


class RankFilter(logging.Filter):
    """过滤器：只允许主进程输出日志"""
    def filter(self, record):
        return dist_utils.is_main_process()


def setup_logger(output_dir):
    """设置logger，确保只有主进程输出日志，避免多GPU模式下重复输出
    
    Args:
        output_dir: 日志文件保存目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    
    # 清除已有的handlers，避免重复添加
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # 创建控制台处理器（带颜色）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s"))
    # 添加过滤器，只允许主进程输出到控制台
    console_handler.addFilter(RankFilter())
    
    # 创建文件处理器（不带颜色，避免 ANSI 代码写入文件）
    # 文件处理器也只让主进程写入
    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    file_handler.addFilter(RankFilter())
    
    # 设置日志级别
    root_logger.setLevel(logging.INFO if dist_utils.is_main_process() else logging.WARN)
    
    # 添加handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # 禁止日志传播到父logger，避免重复输出
    root_logger.propagate = False


def log_on_main(msg, level=logging.INFO):
    """仅在主进程输出日志的便捷函数
    
    Args:
        msg: 日志消息
        level: 日志级别，默认INFO
    
    Example:
        from larm.common.logger import log_on_main
        log_on_main("Training started")
    """
    if dist_utils.is_main_process():
        logging.log(level, msg)


def info_on_main(msg):
    """仅在主进程输出INFO日志"""
    log_on_main(msg, logging.INFO)


def warning_on_main(msg):
    """仅在主进程输出WARNING日志"""
    log_on_main(msg, logging.WARNING)


def error_on_main(msg):
    """仅在主进程输出ERROR日志"""
    log_on_main(msg, logging.ERROR)