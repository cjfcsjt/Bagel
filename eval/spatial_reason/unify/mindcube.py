# --------------------------------------------------------
# MindCube Dataset Loader
# Adapted from SPAR dataset loader for MindCube spatial reasoning benchmark
# --------------------------------------------------------

import os
import io
import random
import argparse
try:
    import orjson as json
except:
    import json

from tqdm import tqdm
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt

import torch


class MindCubeDataset(torch.utils.data.Dataset):
    """
    Dataset for MindCube spatial reasoning benchmark.
    
    Reads a JSONL file where each line is a JSON object with fields:
        - id: sample identifier
        - input_prompt: the text prompt / question
        - images: list of relative image paths
        - gt_answer: ground truth answer
        - category, type, question, etc. (optional metadata)
    """

    def __init__(self, data_file, image_root="./data/", repeat_time=1):
        """
        Args:
            data_file: Path to JSONL file containing the dataset.
            image_root: Root directory for resolving image paths.
            repeat_time: Repeat factor for the data (< 1 for subset, > 1 for repeat).
        """
        print(f"Loading data from {data_file}...")
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))

        if repeat_time < 1:
            self.data = self.data[:int(len(self.data) * repeat_time)]
        if repeat_time > 1:
            assert isinstance(repeat_time, int)
            self.data = self.data * repeat_time

        self.image_root = image_root
        print(f"Loaded {len(self.data)} samples (repeat_time={repeat_time})")

    def __len__(self):
        return len(self.data)

    def resolve_image_path(self, img_path):
        """
        Resolve an image path to an absolute path using image_root.
        Handles relative paths, absolute paths, and Windows-style paths.
        """
        if img_path.startswith(self.image_root):
            return img_path

        # Handle absolute paths (Unix or Windows)
        if os.path.isabs(img_path) or (len(img_path) > 2 and img_path[1] == ':'):
            if "MindCube_image/" in img_path:
                relative_part = img_path.split("MindCube_image/", 1)[1]
            elif "other_all_image/" in img_path:
                relative_part = "other_all_image/" + img_path.split("other_all_image/", 1)[1]
            else:
                relative_part = os.path.basename(img_path)
        else:
            # Relative path - use as is
            relative_part = img_path

        return os.path.join(self.image_root, relative_part)

    def load_image(self, image_path):
        return Image.open(image_path).convert('RGB')

    def __getitem__(self, idx):
        data_item = self.data[idx]

        sample_id = data_item.get('id', idx)
        prompt = data_item.get('input_prompt', '')
        gt_answer = data_item.get('gt_answer', '')
        raw_image_paths = data_item.get('images', [])

        # Resolve and validate image paths
        images = []
        valid_image_paths = []
        for img_path in raw_image_paths:
            resolved = self.resolve_image_path(img_path)
            if os.path.exists(resolved):
                images.append(self.load_image(resolved))
                valid_image_paths.append(resolved)
            else:
                print(f"Warning: Image not found: {resolved}")

        ret = dict(
            images=images,
            question=prompt,
            answer=gt_answer,
            index=sample_id,
            image_paths=valid_image_paths,
            # Keep extra metadata for downstream evaluation
            category=data_item.get('category', []),
            data_type=data_item.get('type', ''),
            raw_data=data_item,
        )
        return ret


def collate_fn(batches, tokenizer=None):
    images = [_['images'] for _ in batches]
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    indices = [_['index'] for _ in batches]
    return images, questions, answers, indices


def build_dataset(data_file, image_root="./data/", repeat_time=1):
    """
    Build a MindCube dataset from a single JSONL file.
    
    Args:
        data_file: Path to the JSONL data file.
        image_root: Root directory for images.
        repeat_time: Repeat factor.
        
    Returns:
        MindCubeDataset instance.
    """
    dataset = MindCubeDataset(
        data_file=data_file,
        image_root=image_root,
        repeat_time=repeat_time,
    )
    print(f"Dataset has {len(dataset)} samples.")
    return dataset


def main(args: argparse.Namespace):
    random.seed(args.seed)

    dataset = build_dataset(
        data_file=args.data_file,
        image_root=args.image_root,
        repeat_time=args.repeat_time,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=None),
    )

    for _, (images, questions, answers, indices) in tqdm(enumerate(dataloader)):
        imgs = images[0]
        question = questions[0]
        answer = answers[0]
        index = indices[0]

        print(f"\nSample #{index}")
        print("🟡 Question:\n", question)
        print("✅ GT Answer:\n", answer)

        num_views = len(imgs)
        if num_views == 0:
            print("⚠️  No valid images for this sample, skipping visualization.")
            continue

        fig, axs = plt.subplots(1, num_views, figsize=(5 * num_views, 5))
        if num_views == 1:
            axs = [axs]

        for i, img in enumerate(imgs):
            axs[i].imshow(img)
            axs[i].set_title(f"View {i + 1}")
            axs[i].axis("off")

        plt.tight_layout()
        os.makedirs(args.image_save_dir, exist_ok=True)
        save_path = os.path.join(args.image_save_dir, f"sample_{index}.png")
        plt.savefig(save_path)
        plt.close(fig)

        print(f"🖼️  Saved visualization to {save_path}")

        user_input = input("Press Enter to continue, or type 'q' to quit: ")
        if user_input.lower() == "q":
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MindCube Dataset Viewer")
    parser.add_argument(
        '--data-file', type=str, required=True,
        help='Path to a JSONL data file (e.g., data/prompts/general/MindCube_tinybench_raw_qa.jsonl)'
    )
    parser.add_argument(
        '--image-root', type=str,
        default='/data/spatial_data/Bagel/eval/spatial_reason/mindcube/MindCube/data/',
        help='Root directory for image files'
    )
    parser.add_argument(
        '--image-save-dir', type=str, default='debug_image_mindcube',
        help='Directory to save visualization images'
    )
    parser.add_argument(
        '--repeat-time', type=int, default=1,
        help='Repeat factor for data (default: 1)'
    )
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    main(args)
