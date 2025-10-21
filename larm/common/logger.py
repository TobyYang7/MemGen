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


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 创建控制台处理器（带颜色）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s"))

    # 创建文件处理器（不带颜色，避免 ANSI 代码写入文件）
    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO if dist_utils.is_main_process() else logging.WARN,
        handlers=[console_handler, file_handler],
    )
