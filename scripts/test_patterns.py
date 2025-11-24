import numpy as np
import torch

def create_test_patterns():
    """创建各种测试模式用于调试"""

    # 1. 单位矩阵模式
    def identity_matrix(size):
        return torch.eye(size, dtype=torch.float16, device='cuda')

    # 2. 序列模式 (便于调试索引)
    def sequence_matrix(rows, cols):
        data = np.arange(rows * cols, dtype=np.float16).reshape(rows, cols)
        return torch.tensor(data, dtype=torch.float16, device='cuda')

    # 3. 棋盘模式
    def checkerboard_matrix(rows, cols):
        data = np.zeros((rows, cols), dtype=np.float16)
        for i in range(rows):
            for j in range(cols):
                data[i, j] = 1.0 if (i + j) % 2 == 0 else 0.0
        return torch.tensor(data, dtype=torch.float16, device='cuda')

    return {
        'identity': identity_matrix,
        'sequence': sequence_matrix,
        'checkerboard': checkerboard_matrix
    }

def analyze_memory_access():
    """分析内存访问模式"""
    patterns = create_test_patterns()

    # 测试小矩阵以验证正确性
    A = patterns['sequence'](4, 4)
    B = patterns['sequence'](4, 4)

    print("Matrix A:")
    print(A.cpu().numpy())
    print("\nMatrix B:")
    print(B.cpu().numpy())

    # 参考结果
    C_ref = torch.matmul(A.float(), B.float())
    print("\nReference result (A @ B):")
    print(C_ref.cpu().numpy())

if __name__ == "__main__":
    analyze_memory_access()