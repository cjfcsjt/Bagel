import os, json, glob, io, argparse
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def ensure_rgb_pil(x):
    # x 可能是 PIL.Image，或 dict/bytes（parquet 常见是 {"bytes":..., "path":...}）
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")

    if isinstance(x, dict):
        b = x.get("bytes", None)
        p = x.get("path", None)
        if b is not None:
            return Image.open(io.BytesIO(b)).convert("RGB")
        if p is not None and os.path.exists(p):
            return Image.open(p).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(x)} keys={list(x.keys()) if isinstance(x, dict) else ''}")

def normalize_conversations(conv):
    """
    期望输出: [{"from":"human","value":"<image>\\n..."}, {"from":"gpt","value":"..."}]
    有些 parquet 里可能是 role/content 或字符串 json，这里做尽量兼容。
    """
    if conv is None:
        return None
    if isinstance(conv, str):
        conv = json.loads(conv)

    out = []
    for m in conv:
        if not isinstance(m, dict):
            continue
        if "from" in m and "value" in m:
            out.append({"from": m["from"], "value": m["value"]})
            continue
        # 兼容 role/content
        role = m.get("role", m.get("speaker", None))
        content = m.get("content", m.get("text", None))
        if role is None or content is None:
            continue
        if role in ("user", "human"):
            out.append({"from": "human", "value": content})
        else:
            out.append({"from": "gpt", "value": content})

    return out if out else None

def ensure_has_image_placeholder(conv):
    # BAGEL 会在对话里找 <image>，找不到会报错/跳过 :contentReference[oaicite:1]{index=1}
    if any(c.get("from") == "human" and "<image>" in c.get("value", "") for c in conv):
        return conv
    for c in conv:
        if c.get("from") == "human":
            c["value"] = "<image>\n" + c.get("value", "")
            break
    return conv

def _save_image(pil_img, path, quality=85):
    """保存单张图片，供线程池调用"""
    pil_img.save(path, quality=quality, optimize=False, subsampling=0)
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="包含 parquet 的目录")
    ap.add_argument("--out_dir", required=True, help="输出目录，会生成 out_dir/vlm/images 和 out_dir/vlm/llava_ov_si.jsonl")
    ap.add_argument("--max_samples", type=int, default=1000, help="先导出多少条测试，0 表示尽量全量（不建议第一次就全量）")
    ap.add_argument("--pattern", default="**/*.parquet", help="parquet glob pattern")
    ap.add_argument("--workers", type=int, default=16, help="图片保存并行线程数")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality (降低可加速，默认85)")
    args = ap.parse_args()

    parquet_files = sorted(glob.glob(os.path.join(args.src_dir, args.pattern), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {args.src_dir} with pattern {args.pattern}")

    vlm_dir = os.path.join(args.out_dir, "vlm")
    img_dir = os.path.join(vlm_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    jsonl_path = os.path.join(vlm_dir, "llava_ov_si.jsonl")

    # 用 datasets 直接读 parquet（流式，不用一次性载入内存）
    ds = load_dataset("parquet", data_files=parquet_files, split="train", streaming=True)

    # 从 parquet 元数据快速统计总行数（只读 footer，不加载数据）
    total_rows = sum(pq.read_metadata(f).num_rows for f in parquet_files)
    print(f"Total rows in parquet files: {total_rows}")
    bar_total = min(args.max_samples, total_rows) if args.max_samples else total_rows

    written = 0
    skipped = 0
    pbar = tqdm(desc="Processing", total=bar_total,
                unit="sample", dynamic_ncols=True)

    # 用线程池并行保存图片（JPEG 编码 + 磁盘写入是主要瓶颈）
    executor = ThreadPoolExecutor(max_workers=args.workers)
    pending_futures = []
    FLUSH_INTERVAL = 200  # 每 200 条 flush 一次，减少 IO 开销

    with open(jsonl_path, "w", encoding="utf-8", buffering=1 << 20) as f:  # 1MB 写缓冲
        for i, ex in enumerate(ds):
            if args.max_samples and args.max_samples > 0 and written >= args.max_samples:
                break

            conv = normalize_conversations(ex.get("conversations", ex.get("conversation", ex.get("messages", None))))
            if conv is None:
                skipped += 1
                continue
            conv = ensure_has_image_placeholder(conv)

            # 取 image 字段（有些样本可能是视频类，这里先跳过）
            if "image" not in ex or ex["image"] is None:
                skipped += 1
                continue

            # 单图（跳过多图样本）
            img_field = ex["image"]
            if isinstance(img_field, list):
                skipped += 1
                continue

            sample_id = ex.get("id", i)
            pil = ensure_rgb_pil(img_field)
            name = f"{sample_id}.jpg"
            fut = executor.submit(_save_image, pil, os.path.join(img_dir, name), args.quality)
            pending_futures.append(fut)
            obj = {"id": sample_id, "image": name, "conversations": conv}

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
            pbar.update(1)
            pbar.set_postfix(written=written, skipped=skipped, scanned=i + 1)

            # 定期回收完成的 future，防止内存堆积 & 及早发现异常
            if len(pending_futures) >= args.workers * 4:
                done = [ft for ft in pending_futures if ft.done()]
                for ft in done:
                    ft.result()  # 如果保存出错，这里会抛异常
                    pending_futures.remove(ft)

    # 等待所有图片保存完成
    pbar.set_description("Waiting for image saves to finish")
    for ft in as_completed(pending_futures):
        ft.result()
    executor.shutdown(wait=True)
    pbar.close()

    print("DONE")
    print("Wrote samples:", written)
    print("Images dir   :", img_dir)
    print("JSONL path   :", jsonl_path)

if __name__ == "__main__":
    main()