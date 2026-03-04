# --------------------------------------------------------
# OST-Bench Dataset Loader
# Adapted for unified evaluation from OST-Bench interleave llava format
#
# Reads: OST_bench_train_interleave_llava.json
#   - Already flattened: multi-round history baked into a single conversation
#   - Each sample has <image> tags in the prompt matching image_paths count
# --------------------------------------------------------

import os
import random
import argparse
import json

from tqdm import tqdm
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt

import torch


# ============================================================================
#                       Accuracy Stats Tracker
# ============================================================================

def blank_stats():
    """
    Create an empty accuracy tracker for OST-Bench task taxonomy.

    Two top-level categories:
        A  = Appearance-based (object existence, quantity, diversity, order, orientation, position, room-size)
        AO = Appearance+Object-based (direction, distance)

    Each has sub-types like "object-existence(option)", "direction(float)", etc.
    """
    return {
        "Total": [],
        # Appearance-based
        "A": {
            "object-existence(option)": [], "object-existence(int1)": [], "object-existence(int2)": [],
            "object-diversity(option)": [], "object-quantity(int)": [],
            "object-order(option)": [],
            "orientation(float)": [], "orientation(option)": [],
            "position(float)": [], "position(option)": [],
            "room-size(float)": [],
            "Total": [],
        },
        # Appearance+Object-based
        "AO": {
            "direction(float)": [], "direction(int)": [],
            "direction(option1)": [], "direction(option2)": [], "direction(option3)": [],
            "distance(float)": [], "distance(int)": [],
            "distance(option1)": [], "distance(option2)": [], "distance(option3)": [],
            "Total": [],
        },
    }


def parse_type(type_str):
    """
    Parse OST-Bench type string into (category, sub_type).
    e.g. "A_object-existence(option)" -> ("A", "object-existence(option)")
         "AO_direction(float)"        -> ("AO", "direction(float)")
    """
    if type_str.startswith("AO_"):
        return "AO", type_str[3:]
    elif type_str.startswith("A_"):
        return "A", type_str[2:]
    else:
        return type_str, type_str


def print_stats(result):
    """Pretty-print per-category and overall accuracy."""
    eps = 1e-6
    overall = sum(result["Total"]) / (len(result["Total"]) + eps) * 100
    print("\n======= FINAL =======")
    print(f"Overall: {overall:.2f}% (N={len(result['Total'])})")
    for cat in [k for k in result if k != "Total"]:
        cat_total = result[cat]["Total"]
        cat_acc = sum(cat_total) / (len(cat_total) + eps) * 100
        print(f"{cat}: {cat_acc:.2f}% (N={len(cat_total)})")
        for sub in sorted(result[cat].keys()):
            if sub == "Total":
                continue
            sub_total = result[cat][sub]
            if len(sub_total) == 0:
                continue
            sub_acc = sum(sub_total) / (len(sub_total) + eps) * 100
            print(f"    {sub}: {sub_acc:.2f}% (N={len(sub_total)})")


# ============================================================================
#                          OST-Bench Dataset
# ============================================================================

class OSTBenchDataset(torch.utils.data.Dataset):
    """
    Dataset for OST-Bench spatial reasoning benchmark (interleave llava format).

    Reads a JSON list file where each element has:
        - id: int
        - type: e.g. "A_object-existence(option)", "AO_direction(float)"
        - prompt_id: unique prompt identifier
        - image_paths: list of relative image paths (under image_root)
        - conversations: [{from: "human", value: "..."}, {from: "gpt", value: ["answer"]}]
            The human value contains <image> tags matching len(image_paths).
    """

    def __init__(self, data_file, image_root, repeat_time=1):
        """
        Args:
            data_file: Path to the JSON file (e.g. OST_bench_train_interleave_llava.json).
            image_root: Root directory for resolving image paths (e.g. .../img_train/).
            repeat_time: Repeat factor for the data.
        """
        print(f"Loading data from {data_file}...")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if repeat_time < 1:
            self.data = self.data[:int(len(self.data) * repeat_time)]
        if repeat_time > 1:
            assert isinstance(repeat_time, int)
            self.data = self.data * repeat_time

        self.image_root = image_root
        print(f"Loaded {len(self.data)} samples (repeat_time={repeat_time})")

    def __len__(self):
        return len(self.data)

    def load_image(self, image_path):
        return Image.open(image_path).convert('RGB')

    def __getitem__(self, idx):
        item = self.data[idx]

        sample_id = item['id']
        type_str = item['type']
        prompt_id = item.get('prompt_id', '')
        category, sub_type = parse_type(type_str)

        # Extract prompt (human turn) and answer (gpt turn)
        conversations = item['conversations']
        prompt = conversations[0]['value']  # human message with <image> tags
        gt_answer_raw = conversations[1]['value']  # list like ["No"] or ["1.46"]
        gt_answer = gt_answer_raw[0] if isinstance(gt_answer_raw, list) else str(gt_answer_raw)

        # Resolve image paths
        raw_image_paths = item.get('image_paths', [])
        images = []
        valid_image_paths = []
        for img_path in raw_image_paths:
            full_path = os.path.join(self.image_root, img_path)
            if os.path.exists(full_path):
                images.append(self.load_image(full_path))
                valid_image_paths.append(full_path)
            else:
                print(f"Warning: Image not found: {full_path}")

        ret = dict(
            images=images,
            image_paths=valid_image_paths,
            question=prompt,
            answer=gt_answer,
            index=sample_id,
            prompt_id=prompt_id,
            type=type_str,
            category=category,
            sub_type=sub_type,
            raw_data=item,
        )
        return ret


def collate_fn(batches, tokenizer=None):
    images = [_['images'] for _ in batches]
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    indices = [_['index'] for _ in batches]
    categories = [_['category'] for _ in batches]
    sub_types = [_['sub_type'] for _ in batches]
    return images, questions, answers, indices, categories, sub_types


def build_dataset(data_file, image_root, repeat_time=1):
    """
    Build an OST-Bench dataset.

    Args:
        data_file: Path to the interleave llava JSON file.
        image_root: Root directory for images.
        repeat_time: Repeat factor.

    Returns:
        OSTBenchDataset instance.
    """
    dataset = OSTBenchDataset(
        data_file=data_file,
        image_root=image_root,
        repeat_time=repeat_time,
    )
    print(f"Dataset has {len(dataset)} samples.")
    return dataset


# ============================================================================
#                              Main (Viewer)
# ============================================================================

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

    for _, (images, questions, answers, indices, categories, sub_types) in tqdm(enumerate(dataloader)):
        imgs = images[0]
        question = questions[0]
        answer = answers[0]
        index = indices[0]
        category = categories[0]
        sub_type = sub_types[0]

        print(f"\nSample #{index}  [{category} / {sub_type}]")
        print(f"🟡 Question:\n{question[:500]}...")
        print(f"✅ GT Answer: {answer}")
        print(f"📷 Num images: {len(imgs)}")

        if len(imgs) == 0:
            print("⚠️  No valid images for this sample, skipping visualization.")
            continue

        # Visualize (show up to 8 images; if more, show first and last 4)
        num_views = len(imgs)
        if num_views <= 8:
            show_imgs = imgs
            show_titles = [f"Img {i+1}" for i in range(num_views)]
        else:
            show_imgs = imgs[:4] + imgs[-4:]
            show_titles = [f"Img {i+1}" for i in range(4)] + [f"Img {num_views-3+i}" for i in range(4)]

        n_show = len(show_imgs)
        fig, axs = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
        if n_show == 1:
            axs = [axs]

        for i, img in enumerate(show_imgs):
            axs[i].imshow(img)
            axs[i].set_title(show_titles[i])
            axs[i].axis("off")

        plt.suptitle(f"[{category}/{sub_type}] Total: {num_views} images", fontsize=10)
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
    parser = argparse.ArgumentParser(description="OST-Bench Dataset Viewer")
    parser.add_argument(
        '--data-file', type=str,
        default='/data/spatial_data/reason_data/OST-Bench/OST_bench_train_interleave_llava.json',
        help='Path to the interleave llava JSON file'
    )
    parser.add_argument(
        '--image-root', type=str,
        default='/data/spatial_data/reason_data/OST-Bench/img_train/',
        help='Root directory for image files'
    )
    parser.add_argument(
        '--image-save-dir', type=str, default='debug_image_ost',
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
