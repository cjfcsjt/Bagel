# --------------------------------------------------------
# OmniSpatial Dataset Loader
# Adapted from OmniSpatial/vlms_eval/qwenvl_eval.py for unified evaluation
# --------------------------------------------------------

import os
import re
import random
import argparse
import json

from tqdm import tqdm
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt

import torch


# ============================================================================
#                          System & Format Prompts
# ============================================================================

DEFAULT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive  
1. **Image** - a single RGB frame depicting a scene.  
2. **Question** - a natural-language query about spatial relationships between objects in the image.  
3. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Based on the image and question, provide your answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply "insufficient information."
"""

ZERO_SHOT_COT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive  
1. **Image** - a single RGB frame depicting a scene.
2. **Question** - a natural-language query about spatial relationships between objects in the image.
3. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Think step by step and provide the answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply "insufficient information."
"""

MANUAL_COT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive  
1. **Image** - a single RGB frame depicting a scene.  
2. **Question** - a natural-language query about spatial relationships between objects in the image.  
3. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Guidelines
----------
Please follow these steps to analyze the image and answer the question:
1. First, carefully observe the image and identify all relevant objects and their spatial relationships.
2. Next, break down the question into key components that need to be addressed.
3. Think through the spatial reasoning step-by-step to arrive at your answer. It may be necessary to transfer perspective to better understand the scene.
4. Finally, select the most appropriate option (A, B, C, or D) based on your analysis.

Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply "insufficient information."
"""

SYS_PROMPTS = {
    "none": DEFAULT_SYSTEM_PROMPT,
    "zeroshot_cot": ZERO_SHOT_COT_SYSTEM_PROMPT,
    "manual_cot": MANUAL_COT_SYSTEM_PROMPT,
}

RE_FORMAT = """
End your answer with a separate line formatted exactly as:

Answer: X
where X ∈ {A, B, C, D}.
"""

JSON_FORMAT = """
You need to respond with the answer in JSON format:

```json
{
    "analysis": "The analysis of the image and question",
    "answer": "A"
}
```
"""

DIRECT_FORMAT = """
Note: You only need to respond with A, B, C, or D without providing any additional information.
"""

FORMAT_PROMPTS = {
    "re": RE_FORMAT,
    "json": JSON_FORMAT,
    "direct": DIRECT_FORMAT,
}


# ============================================================================
#                          Accuracy Stats Tracker
# ============================================================================

def blank_stats():
    """Create an empty accuracy tracker matching OmniSpatial task taxonomy."""
    return {
        "Total": [],
        "Dynamic_Reasoning": {"Manipulation": [], "Motion_Analysis": [], "Total": []},
        "Spatial_Interaction": {"Traffic_Analysis": [], "Localization": [], "Geospatial_Strategy": [], "Total": []},
        "Complex_Logic": {"Pattern_Recognition": [], "Geometric_Reasoning": [], "Total": []},
        "Perspective_Taking": {"Egocentric": [], "Allocentric": [], "Hypothetical": [], "Total": []},
    }


def print_stats(result):
    """Pretty-print per-task and overall accuracy."""
    eps = 1e-6
    overall = sum(result["Total"]) / (len(result["Total"]) + eps) * 100
    print("\n======= FINAL =======")
    print(f"Overall: {overall:.2f}% (N={len(result['Total'])})")
    for task in [k for k in result if k not in {"Total"}]:
        task_total = result[task]["Total"]
        task_acc = sum(task_total) / (len(task_total) + eps) * 100
        print(f"{task}: {task_acc:.2f}%")
        for sub in result[task]:
            if sub == "Total":
                continue
            sub_total = result[task][sub]
            sub_acc = sum(sub_total) / (len(sub_total) + eps) * 100
            print(f"    {sub}: {sub_acc:.2f}%")


# ============================================================================
#                         OmniSpatial Dataset
# ============================================================================

class OmniSpatialDataset(torch.utils.data.Dataset):
    """
    Dataset for OmniSpatial spatial reasoning benchmark.

    Reads a data.json file (JSON list) where each element has:
        - id: e.g. "0_0"
        - question: text question
        - options: list of option strings
        - answer: int index of the correct option (0-based)
        - task_type: e.g. "Dynamic_Reasoning"
        - sub_task_type: e.g. "Motion_Analysis"

    Image is located at: {dataset_path}/{task_type}/{id.split('_')[0]}.png
    """

    def __init__(self, dataset_path, prompt_type="manual_cot", eval_type="re", repeat_time=1):
        """
        Args:
            dataset_path: Root directory of the OmniSpatial dataset (contains data.json + task folders).
            prompt_type: System prompt style ("none", "zeroshot_cot", "manual_cot").
            eval_type: Answer extraction format ("re", "json", "direct").
            repeat_time: Repeat factor for the data.
        """
        data_file = os.path.join(dataset_path, 'data.json')
        print(f"Loading data from {data_file}...")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if repeat_time < 1:
            self.data = self.data[:int(len(self.data) * repeat_time)]
        if repeat_time > 1:
            assert isinstance(repeat_time, int)
            self.data = self.data * repeat_time

        self.dataset_path = dataset_path
        self.prompt_type = prompt_type
        self.eval_type = eval_type
        print(f"Loaded {len(self.data)} samples (repeat_time={repeat_time})")

    def __len__(self):
        return len(self.data)

    def get_image_path(self, info):
        """Resolve image path from sample info: {dataset_path}/{task_type}/{id_prefix}.png"""
        raw_id = info["id"]
        task_type = info["task_type"]
        return os.path.join(self.dataset_path, task_type, f"{raw_id.split('_')[0]}.png")

    def build_prompt(self, info):
        """
        Build the full text prompt including system prompt, format instruction,
        question, and options — matching qwenvl_eval.py behavior.
        """
        question = info["question"]
        options = info["options"]

        prompt = SYS_PROMPTS[self.prompt_type] + '\n' + FORMAT_PROMPTS[self.eval_type] + '\n\n' + question
        for i, opt in enumerate(options):
            prompt += f"\n{chr(65 + i)}. {opt}"
        return prompt

    def load_image(self, image_path):
        return Image.open(image_path).convert('RGB')

    def __getitem__(self, idx):
        info = self.data[idx]

        sample_id = info["id"]
        gt_answer_idx = info["answer"]  # int index
        gt_letter = chr(65 + gt_answer_idx)  # "A", "B", "C", "D"
        task_type = info["task_type"]
        sub_task_type = info["sub_task_type"]

        # Build prompt
        prompt = self.build_prompt(info)

        # Load image
        image_path = self.get_image_path(info)
        images = []
        if os.path.exists(image_path):
            images.append(self.load_image(image_path))
        else:
            print(f"Warning: Image not found: {image_path}")

        ret = dict(
            images=images,
            image_paths=[image_path],
            question=prompt,
            answer=gt_letter,
            index=sample_id,
            task_type=task_type,
            sub_task_type=sub_task_type,
            options=info["options"],
            raw_data=info,
        )
        return ret


def collate_fn(batches, tokenizer=None):
    images = [_['images'] for _ in batches]
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    indices = [_['index'] for _ in batches]
    task_types = [_['task_type'] for _ in batches]
    sub_task_types = [_['sub_task_type'] for _ in batches]
    return images, questions, answers, indices, task_types, sub_task_types


def build_dataset(dataset_path, prompt_type="manual_cot", eval_type="re", repeat_time=1):
    """
    Build an OmniSpatial dataset.

    Args:
        dataset_path: Root directory of the OmniSpatial dataset.
        prompt_type: System prompt style.
        eval_type: Answer extraction format.
        repeat_time: Repeat factor.

    Returns:
        OmniSpatialDataset instance.
    """
    dataset = OmniSpatialDataset(
        dataset_path=dataset_path,
        prompt_type=prompt_type,
        eval_type=eval_type,
        repeat_time=repeat_time,
    )
    print(f"Dataset has {len(dataset)} samples.")
    return dataset


# ============================================================================
#                     Answer Extraction from Response
# ============================================================================

def extract_answer(response, eval_type="re"):
    """
    Extract the predicted answer letter from model response.

    Args:
        response: Raw model response string.
        eval_type: Extraction method ("re", "json", "direct").

    Returns:
        Predicted answer letter (e.g. "A").
    """
    if eval_type == "json":
        try:
            cleaned = response.strip().removeprefix("```json").removesuffix("```").strip()
            pred_letter = json.loads(cleaned).get("answer", "A").strip().upper()[:1]
        except Exception:
            pred_letter = "A"
    elif eval_type == "re":
        pattern = re.compile(r"Answer\s*:\s*([A-D])\b", re.IGNORECASE)
        matches = pattern.findall(response)
        pred_letter = matches[-1] if matches else "A"
    elif eval_type == "direct":
        pred_letter = response.strip().upper()[:1]
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")
    return pred_letter


# ============================================================================
#                              Main (Viewer)
# ============================================================================

def main(args: argparse.Namespace):
    random.seed(args.seed)

    dataset = build_dataset(
        dataset_path=args.dataset_path,
        prompt_type=args.prompt_type,
        eval_type=args.eval_type,
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

    result = blank_stats()

    for _, (images, questions, answers, indices, task_types, sub_task_types) in tqdm(enumerate(dataloader)):
        imgs = images[0]
        question = questions[0]
        answer = answers[0]
        index = indices[0]
        task_type = task_types[0]
        sub_task_type = sub_task_types[0]

        print(f"\nSample #{index}  [{task_type} / {sub_task_type}]")
        print("🟡 Question:\n", question)
        print("✅ GT Answer:", answer)

        if len(imgs) == 0:
            print("⚠️  No valid images for this sample, skipping visualization.")
            continue

        # Visualize
        num_views = len(imgs)
        fig, axs = plt.subplots(1, num_views, figsize=(5 * num_views, 5))
        if num_views == 1:
            axs = [axs]

        for i, img in enumerate(imgs):
            axs[i].imshow(img)
            axs[i].set_title(f"Image")
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
    parser = argparse.ArgumentParser(description="OmniSpatial Dataset Viewer")
    parser.add_argument(
        '--dataset-path', type=str, required=True,
        help='Root directory of OmniSpatial dataset (contains data.json and task folders)'
    )
    parser.add_argument(
        '--prompt-type', type=str, default='manual_cot',
        choices=['none', 'zeroshot_cot', 'manual_cot'],
        help='System prompt style'
    )
    parser.add_argument(
        '--eval-type', type=str, default='re',
        choices=['re', 'json', 'direct'],
        help='Answer extraction format'
    )
    parser.add_argument(
        '--image-save-dir', type=str, default='debug_image_omnispatial',
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
