import sys
import os
import threading
from loguru import logger
from typing import Optional

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | [{extra[module]: <12}] {message}"
LOG_FORMAT_FILE = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | [{extra[module]: <12}] {message} | {file}:{line}"

_initialized = False


class  TraceSampler:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, enabled: bool = True, sample_interval: int = 30):
        if self._initialized:
            return
        self.enabled = enabled
        self.sample_interval = sample_interval
        self._counters = {}
        self._counter_lock = threading.Lock()
        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> "TraceSampler":
        if cls._instance is None:
            return cls(enabled=True, sample_interval=30)
        return cls._instance
    
    def should_log(self, category: str = "default") -> bool:
        if not self.enabled:
            return False
        
        with self._counter_lock:
            if category not in self._counters:
                self._counters[category] = 0
            self._counters[category] += 1
            return self._counters[category] % self.sample_interval == 0
    
    def reset(self):
        with self._counter_lock:
            self._counters.clear()


def init_trace_sampler(enabled: bool = True, sample_interval: int = 30):
    TraceSampler(enabled=enabled, sample_interval=sample_interval)

def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    file_enabled: bool = True,
    console_enabled: bool = True,
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "zip",
):
    global _initialized
    
    if _initialized:
        return
    
    logger.remove()
    
    logger.configure(extra={"module": "system"})
    
    if console_enabled:
        logger.add(
            sys.stderr,
            level=level,
            format=LOG_FORMAT,
            colorize=True,
            enqueue=True,
        )
    
    if file_enabled:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_path = os.path.join(log_dir, "human_detection_{time:YYYY-MM-DD}.log")
        logger.add(
            log_path,
            level=level,
            format=LOG_FORMAT_FILE,
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True,
            encoding="utf-8",
        )
    
    _initialized = True
    logger.bind(module="logger").success("Logging system initialized")


def get_logger(module: str):
    return logger.bind(module=module)


def log_function_entry(log, func_name: str, **kwargs):
    params = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    log.debug(f"[{func_name}] >>> START" + (f" | {params}" if params else ""))


def log_function_exit(log, func_name: str, result: str = ""):
    log.debug(f"[{func_name}] <<< END" + (f" | {result}" if result else ""))


class PerformanceLogger:
    def __init__(self, log, name: str, use_sampler: bool = True):
        self.log = log
        self.name = name
        self.start_time = None
        self.use_sampler = use_sampler
        self._sampler = TraceSampler.get_instance() if use_sampler else None
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        if self._sampler is None or self._sampler.should_log("perf"):
            self.log.trace(f"[{self.name}] >>> START")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = (time.perf_counter() - self.start_time) * 1000
        if exc_type is not None:
            self.log.error(f"[{self.name}] <<< FAILED | error={exc_val}")
        elif self._sampler is None or self._sampler.should_log("perf"):
            self.log.trace(f"[{self.name}] <<< END | elapsed={elapsed:.2f}ms")
        return False
