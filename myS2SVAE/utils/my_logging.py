# -*- coding:utf-8 -*-
import logging

def create_logger(log_file_name):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s: %(levelname)-8s |%(name)-16s  |%(message)s',
                        datefmt='%y-%m-%d %H:%M:%S',
                        filename=log_file_name,
                        filemode='w')

    # 定义一个Handler打印INFO及以上级别的日志到sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.WARN)
    # 设置日志打印格式
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # 将定义好的console日志handler添加到root logger
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger('logger')


    logging.info("Create logger sucessfully.")
    return logger


if __name__ == '__main__':
    logger = create_logger("my_log.txt")
    logger.debug("Debug message.")
    logger.warning("warning message.")
    logger.info("info message.")
    logger.error("error message.")

