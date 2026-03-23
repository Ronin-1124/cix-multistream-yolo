import os
import numpy as np
import logging
from libnoe import (
    NPU,
    NOE_TENSOR_TYPE_INPUT,
    NOE_TENSOR_TYPE_OUTPUT,
    noe_create_job_cfg_t,
    noe_data_type_t,
)


def get_data_type_info(d_type) -> tuple:
    """获取数据类型信息"""
    if d_type == noe_data_type_t.NOE_DATA_TYPE_S8:
        return (np.int8, -128, 127, np.int8)
    elif d_type == noe_data_type_t.NOE_DATA_TYPE_U8:
        return (np.uint8, 0, 255, np.uint8)
    elif d_type == noe_data_type_t.NOE_DATA_TYPE_S16:
        return (np.int16, -32768, 32767, np.int16)
    elif d_type == noe_data_type_t.NOE_DATA_TYPE_U16:
        return (np.uint16, 0, 65535, np.uint16)
    elif d_type == noe_data_type_t.NOE_DATA_TYPE_S32:
        return (np.int32, -2147483648, 2147483647, np.int32)
    elif d_type == noe_data_type_t.NOE_DATA_TYPE_U32:
        return (np.uint32, 0, 4294967295, np.uint32)
    elif d_type == noe_data_type_t.NOE_DATA_TYPE_F16:
        return (np.float16, 0, 0, np.float16)
    else:
        raise NotImplementedError(f"不支持的数据类型: {d_type}")


class InferenceEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path

        self.job_cfg = noe_create_job_cfg_t()
        self.npu = NPU()

        self.input_type = []
        self.input_dtype_min = []
        self.input_dtype_max = []
        self.in_tensor_desc = []

        self.output_type = []
        self.output_dtype_min = []
        self.output_dtype_max = []
        self.out_tensor_desc = []

        self._init_context()
        self._load_graph()
        self._setup_tensors(NOE_TENSOR_TYPE_INPUT)
        self._setup_tensors(NOE_TENSOR_TYPE_OUTPUT)
        self._create_job()

    def _init_context(self):
        if self.npu.noe_init_context() != 0:
            raise RuntimeError("NPU初始化失败: noe_init_context")
        print(f"[PID {os.getpid()}] NPU上下文初始化成功")

    def _load_graph(self):
        ret, graph_id = self.npu.noe_load_graph(self.model_path)
        if ret != 0:
            raise RuntimeError(f"加载模型失败: {self.model_path}")
        self.graph_id = graph_id
        print(f"[PID {os.getpid()}] 模型加载成功, graph_id={graph_id}")

    def _get_tensor_count(self, tensor_type: int) -> int:
        ret, count = self.npu.noe_get_tensor_count(self.graph_id, tensor_type)
        if ret != 0:
            raise RuntimeError(f"获取tensor数量失败, type={tensor_type}")
        return count

    def _setup_tensors(self, tensor_type: int):
        tensor_count = self._get_tensor_count(tensor_type)
        tensor_list = (
            self.in_tensor_desc
            if tensor_type == NOE_TENSOR_TYPE_INPUT
            else self.out_tensor_desc
        )
        props = (
            (self.input_type, self.input_dtype_min, self.input_dtype_max)
            if tensor_type == NOE_TENSOR_TYPE_INPUT
            else (self.output_type, self.output_dtype_min, self.output_dtype_max)
        )

        for idx in range(tensor_count):
            desc = self.npu.noe_get_tensor_descriptor(self.graph_id, tensor_type, idx)
            tensor_list.append(desc)
            dtype_info = get_data_type_info(desc.data_type)
            for prop, val in zip(props, dtype_info[:3]):
                prop.append(val)

    def _create_job(self):
        ret, job_id = self.npu.noe_create_job(self.graph_id, self.job_cfg)
        if ret != 0:
            raise RuntimeError("创建推理任务失败")
        self.job_id = job_id
        print(f"[PID {os.getpid()}] 推理任务创建成功, job_id={job_id}")

    def _quantize_input(self, data: np.ndarray, tensor_idx: int) -> np.ndarray:
        """量化输入数据"""
        quantized = np.round(
            data.astype(float) * self.in_tensor_desc[tensor_idx].scale
            - self.in_tensor_desc[tensor_idx].zero_point
        )
        quantized = np.clip(
            quantized,
            self.input_dtype_min[tensor_idx],
            self.input_dtype_max[tensor_idx],
        ).astype(self.input_type[tensor_idx])
        return quantized

    def _dequantize_output(self, data: bytes, tensor_idx: int) -> np.ndarray:
        """反量化输出数据"""
        out_data = np.frombuffer(data, dtype=self.output_type[tensor_idx])
        return (
            out_data.astype(np.float32) + self.out_tensor_desc[tensor_idx].zero_point
        ) / self.out_tensor_desc[tensor_idx].scale

    def forward(self, input_datas: np.ndarray) -> list:
        """推理前向传播"""
        if not isinstance(input_datas, np.ndarray):
            input_datas = np.array(input_datas)

        job_id = self.job_id

        # 量化并加载输入
        for i in range(len(self.in_tensor_desc)):
            quantized = self._quantize_input(input_datas, i)
            if len(quantized.tobytes()) != self.in_tensor_desc[i].size:
                raise RuntimeError(
                    f"输入数据大小错误: 期望{self.in_tensor_desc[i].size}, 实际{len(quantized.tobytes())}"
                )
            self.npu.noe_load_tensor(job_id, i, quantized.tobytes())

        # 执行推理
        self.npu.noe_job_infer_sync(job_id, -1)

        # 获取输出
        outputs = []
        for j in range(len(self.out_tensor_desc)):
            ret, data_bytes = self.npu.noe_get_tensor(job_id, NOE_TENSOR_TYPE_OUTPUT, j)
            if ret != 0:
                raise RuntimeError("获取输出tensor失败")
            outputs.append(self._dequantize_output(data_bytes, j))

        return outputs

    def clean(self):
        """清理NPU资源"""
        if hasattr(self, "job_id"):
            self.npu.noe_clean_job(self.job_id)
        if hasattr(self, "graph_id"):
            self.npu.noe_unload_graph(self.graph_id)
        self.npu.noe_deinit_context()


if __name__ == "__main__":

    def benchmark_multithread(num_threads, num_runs=10, warmup=2):
        model_path = "models/yolov8n.cix"
        fake_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

        import threading
        import time

        engines = [InferenceEngine(model_path) for _ in range(num_threads)]
        results = [False] * num_threads

        print(f"\n--- 预热 ({warmup}轮) ---")
        for i in range(num_threads):
            for _ in range(warmup):
                engines[i].forward(fake_input)

        print(f"\n--- 测试 {num_threads} 线程, 每线程 {num_runs} 轮推理 ---")

        def worker(engine, fake_input, results, index):
            for _ in range(num_runs):
                engine.forward(fake_input)
            results[index] = True

        threads = []
        start_all = time.perf_counter()
        for i in range(num_threads):
            t = threading.Thread(
                target=worker, args=(engines[i], fake_input, results, i)
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        total_elapsed = time.perf_counter() - start_all

        total_inferences = num_threads * num_runs
        total_fps = total_inferences / total_elapsed

        print(f"总耗时: {total_elapsed:.3f}s")
        print(f"总推理次数: {total_inferences}")
        print(f"总FPS: {total_fps:.2f}")

        for eng in engines:
            eng.clean()

        return total_fps

    def benchmark_multiprocess(num_processes, num_runs=10, warmup=2):
        model_path = "models/yolov8n.cix"
        fake_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

        import multiprocessing
        import time

        def worker(pid, num_runs, warmup):
            engine = InferenceEngine(model_path)
            for _ in range(warmup):
                engine.forward(fake_input)
            for _ in range(num_runs):
                engine.forward(fake_input)
            engine.clean()

        print(f"\n--- 预热 ({warmup}轮) ---")
        print(f"\n--- 测试 {num_processes} 进程, 每进程 {num_runs} 轮推理 ---")

        processes = []
        start_all = time.perf_counter()
        for i in range(num_processes):
            p = multiprocessing.Process(target=worker, args=(i, num_runs, warmup))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        total_elapsed = time.perf_counter() - start_all

        total_inferences = num_processes * num_runs
        total_fps = total_inferences / total_elapsed

        print(f"总耗时: {total_elapsed:.3f}s")
        print(f"总推理次数: {total_inferences}")
        print(f"总FPS: {total_fps:.2f}")

        return total_fps

    print("=== 多线程测试 ===")
    best_fps = 0
    best_threads = 0
    for num_threads in [1, 2, 3, 4, 5, 6, 7, 8]:
        fps = benchmark_multithread(num_threads=num_threads, num_runs=50, warmup=5)
        if fps > best_fps:
            best_fps = fps
            best_threads = num_threads
    print(f"最佳线程数: {best_threads}, 最高FPS: {best_fps:.2f}")

    print("\n=== 多进程测试 ===")
    best_fps = 0
    best_processes = 0
    for num_processes in [1, 2, 3, 4, 5, 6, 7, 8]:
        fps = benchmark_multiprocess(num_processes=num_processes, num_runs=50, warmup=5)
        if fps > best_fps:
            best_fps = fps
            best_processes = num_processes
    print(f"最佳进程数: {best_processes}, 最高FPS: {best_fps:.2f}")
