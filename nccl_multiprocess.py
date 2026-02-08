#!/usr/bin/env python
"""
Multi-process NCCL Test Script
Usage: torchrun --nproc_per_node=2 nccl_multiprocess_test.py
"""
import torch
import torch.distributed as dist
import os
import sys

def main():
    print("=" * 70)
    print("Multi-Process NCCL Test")
    print("=" * 70)
    
    # 获取环境变量
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"[Rank {rank}] Starting initialization...")
    print(f"[Rank {rank}] LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
    
    # 检查基本信息
    if rank == 0:
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"NCCL version: {'.'.join(map(str, torch.cuda.nccl.version()))}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"NCCL available: {dist.is_nccl_available()}\n")
    
    try:
        # 步骤 1: 设置设备（必须在 init_process_group 之前）
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}] ✓ Set device to cuda:{local_rank}")
        
        # 步骤 2: 初始化进程组
        dist.init_process_group(backend='nccl')
        print(f"[Rank {rank}] ✓ Process group initialized")
        
        # 步骤 3: 验证初始化
        assert dist.get_rank() == rank
        assert dist.get_world_size() == world_size
        print(f"[Rank {rank}] ✓ Rank and world_size verified")
        
        # 步骤 4: 测试 all_reduce
        tensor = torch.ones(1).cuda() * (rank + 1)
        print(f"[Rank {rank}] Before all_reduce: {tensor.item()}")
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(1, world_size + 1))
        print(f"[Rank {rank}] After all_reduce: {tensor.item()} (expected: {expected})")
        
        assert abs(tensor.item() - expected) < 1e-5, f"all_reduce failed: got {tensor.item()}, expected {expected}"
        print(f"[Rank {rank}] ✓ all_reduce test passed!")
        
        # 步骤 5: 测试 broadcast
        if rank == 0:
            broadcast_tensor = torch.tensor([42.0]).cuda()
        else:
            broadcast_tensor = torch.zeros(1).cuda()
        
        dist.broadcast(broadcast_tensor, src=0)
        assert broadcast_tensor.item() == 42.0, f"broadcast failed: got {broadcast_tensor.item()}"
        print(f"[Rank {rank}] ✓ broadcast test passed!")
        
        # 步骤 6: 测试 barrier
        dist.barrier()
        print(f"[Rank {rank}] ✓ barrier test passed!")
        
        # 清理
        dist.destroy_process_group()
        print(f"[Rank {rank}] ✓ Process group destroyed")
        
        if rank == 0:
            print("\n" + "=" * 70)
            print("✅ All NCCL tests passed successfully!")
            print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"[Rank {rank}] ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
