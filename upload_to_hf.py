#!/usr/bin/env python3
"""
Hugging Face Hub 文件上传脚本
用于将 combined.zip 文件上传到 Hugging Face Hub
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo

def upload_to_hf(file_path, repo_id, repo_type="dataset", token=None):
    """
    上传文件到 Hugging Face Hub
    
    Args:
        file_path (str): 要上传的本地文件路径
        repo_id (str): Hugging Face 仓库ID（格式：username/repo-name）
        repo_type (str): 仓库类型，可选 "model", "dataset", "space"
        token (str): Hugging Face token，如果为None则使用环境变量HF_TOKEN
    """
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return False
    
    # 初始化Hugging Face API
    api = HfApi(token=token)
    
    try:
        # 创建仓库（如果不存在）
        create_repo(repo_id, repo_type=repo_type, exist_ok=True, token=token)
        print(f"仓库 {repo_id} 已创建或已存在")
        
        # 上传文件
        print(f"正在上传文件 {file_path}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✅ 文件上传成功！")
        print(f"🔗 访问地址：https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ 上传失败：{e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="上传文件到 Hugging Face Hub")
    parser.add_argument("--file", "-f", default="data/download/combined.zip", 
                       help="要上传的文件路径（默认：data/download/combined.zip）")
    parser.add_argument("--repo", "-r", required=True,
                       help="Hugging Face 仓库ID（格式：username/repo-name）")
    parser.add_argument("--type", "-t", default="dataset", 
                       choices=["model", "dataset", "space"],
                       help="仓库类型（默认：dataset）")
    parser.add_argument("--token", help="Hugging Face token（可选，默认使用环境变量HF_TOKEN）")
    
    args = parser.parse_args()
    
    # 执行上传
    upload_to_hf(args.file, args.repo, args.type, args.token)

if __name__ == "__main__":
    main()