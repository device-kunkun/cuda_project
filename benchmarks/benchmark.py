import torch
import time
import numpy as np
from typing import List, Tuple

def benchmark_torch_matmul(shapes: List[Tuple[int, int, int]], dtype=torch.float16):
    """使用PyTorch作为基准参考"""
    results = []

    for M, N, K in shapes:
        # 创建随机矩阵
        A = torch.randn(M, K, dtype=dtype, device='cuda')
        B = torch.randn(K, N, dtype=dtype, device='cuda')

        # 预热
        for _ in range(10):
            C = torch.matmul(A, B)

        # 基准测试
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(100):
            C = torch.matmul(A, B)

        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒
        tflops = (2 * M * N * K) / (avg_time * 1e9)  # TFLOPS

        results.append({
            'shape': (M, N, K),
            'time_ms': avg_time,
            'tflops': tflops
        })

        print(f"Shape ({M}, {N}, {K}): {avg_time:.3f} ms, {tflops:.2f} TFLOPS")

    return results

if __name__ == "__main__":
    # 测试不同的矩阵大小
    shapes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    print("PyTorch FP16 Matrix Multiplication Benchmark")
    print("=" * 50)
    results = benchmark_torch_matmul(shapes)