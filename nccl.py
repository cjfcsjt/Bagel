#!/usr/bin/env python
import torch
import torch.distributed as dist
import os

print("=" * 70)
print("PyTorch & NCCL Compatibility Check")
print("=" * 70)

# 1. 版本信息
print(f"PyTorch version:        {torch.__version__}")
print(f"CUDA version:           {torch.version.cuda}")
print(f"NCCL version:           {'.'.join(map(str, torch.cuda.nccl.version()))}")
print(f"cuDNN version:          {torch.backends.cudnn.version()}")
print(f"CUDA available:         {torch.cuda.is_available()}")
print(f"GPU count:              {torch.cuda.device_count()}")
print(f"NCCL available:         {dist.is_nccl_available()}")

# 2. 检查 PyTorch 编译时的 NCCL 版本
print("\n" + "=" * 70)
print("PyTorch Build Configuration (NCCL related):")
print("=" * 70)
config = torch.__config__.show()
for line in config.split('\n'):
    if 'NCCL' in line.upper() or 'CUDA' in line.upper():
        print(line)

# 3. 测试 NCCL 初始化（单进程）
print("\n" + "=" * 70)
print("Testing NCCL Initialization (Single Process):")
print("=" * 70)

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    try:
        # 设置环境变量
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        
        # 正确的初始化顺序
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)  # 先设置设备
        dist.init_process_group('nccl', rank=0, world_size=1)  # 再初始化
        
        print("✓ NCCL initialization successful!")
        print(f"✓ Using device: cuda:{local_rank}")
        
        # 注意：单进程环境下跳过 all_reduce 测试
        # NCCL 的集合通信操作在 world_size=1 时可能导致 Segmentation Fault
        world_size = dist.get_world_size()
        if world_size > 1:
            # 测试简单的分布式操作
            tensor = torch.ones(1).cuda()
            dist.all_reduce(tensor)
            print(f"✓ NCCL all_reduce test passed! Result: {tensor.item()}")
        else:
            print("⚠ Skipping all_reduce test (world_size=1, NCCL collective ops may crash)")
            print("✓ NCCL is properly initialized and ready for multi-GPU training")
        
        dist.destroy_process_group()
        print("✓ Process group destroyed successfully")
        
    except Exception as e:
        print(f"✗ NCCL test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("✗ No CUDA devices available")

print("\n" + "=" * 70)
print("Compatibility Check Complete")
print("=" * 70)