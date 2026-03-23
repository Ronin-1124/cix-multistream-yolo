"""
test_inference_pipeline.py

测试推理管道模块。由于 InferenceEngine 依赖 NPU 库 (libnoe)，
测试会在库不可用时跳过相关部分。
"""
import multiprocessing as mp
import threading
import pytest


# 条件导入：如果 NPU 库不可用则跳过
inference = pytest.importorskip("src.processing.inference", reason="NPU library (libnoe) not available")
InferenceEngine = inference.InferenceEngine


class TestInferenceEngineImport:
    """InferenceEngine 导入测试"""

    def test_inference_engine_import(self):
        """InferenceEngine 应该可以被导入"""
        assert InferenceEngine is not None

    def test_inference_engine_init(self):
        """InferenceEngine 初始化需要 model_path"""
        # 这只是一个签名测试，不实际初始化
        import inspect
        sig = inspect.signature(InferenceEngine.__init__)
        params = list(sig.parameters.keys())
        assert 'model_path' in params


class TestInferencePipelineFunctions:
    """推理管道函数存在性测试"""

    def test_inferencing_process_exists(self):
        """inferencing_process 函数应存在"""
        from src.processing.inference_pipeline import inferencing_process
        assert callable(inferencing_process)

    def test_inferencing_thread_exists(self):
        """inferencing_thread 函数应存在"""
        from src.processing.inference_pipeline import inferencing_thread
        assert callable(inferencing_thread)

    def test_inferencing_processes_exists(self):
        """inferencing_processes 函数应存在"""
        from src.processing.inference_pipeline import inferencing_processes
        assert callable(inferencing_processes)

    def test_inferencing_threads_exists(self):
        """inferencing_threads 函数应存在"""
        from src.processing.inference_pipeline import inferencing_threads
        assert callable(inferencing_threads)


class TestInferencePoolFunctions:
    """推理池函数存在性测试"""

    def test_inferencing_threadpool_exists(self):
        """inferencing_threadpool 函数应存在"""
        from src.processing.inference_pool import inferencing_threadpool
        assert callable(inferencing_threadpool)

    def test_inferencing_processpool_exists(self):
        """inferencing_processpool 函数应存在"""
        from src.processing.inference_pool import inferencing_processpool
        assert callable(inferencing_processpool)

    def test_shutdown_threadpool_exists(self):
        """shutdown_threadpool 函数应存在"""
        from src.processing.inference_pool import shutdown_threadpool
        assert callable(shutdown_threadpool)

    def test_shutdown_processpool_exists(self):
        """shutdown_processpool 函数应存在"""
        from src.processing.inference_pool import shutdown_processpool
        assert callable(shutdown_processpool)


class TestMultiprocessingPrimitives:
    """多进程原语测试"""

    def test_queue_creation(self):
        """应该可以创建 multiprocessing.Queue"""
        queue = mp.Queue()
        queue.put("test")
        assert queue.get() == "test"
        queue.close()

    def test_event_creation(self):
        """应该可以创建 multiprocessing.Event"""
        event = mp.Event()
        assert not event.is_set()
        event.set()
        assert event.is_set()
        event.clear()

    def test_stop_event_in_threads(self):
        """threading.Event 应用场景测试"""
        stop_event = threading.Event()
        assert not stop_event.is_set()

        stop_event.set()
        assert stop_event.is_set()

        stop_event.clear()
        assert not stop_event.is_set()
