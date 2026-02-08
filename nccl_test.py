import os
import torch
import torch.distributed as dist

def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    print('local_rank: ', local_rank)
    torch.cuda.set_device(local_rank)
    torch.zeros(1).cuda()
    
    dist.init_process_group(backend='nccl', device_id=torch.device(f'cuda:{local_rank}'))
    print(dist.get_world_size())
    dist.barrier()

if __name__ == '__main__':
    main()