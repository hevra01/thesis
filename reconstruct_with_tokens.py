import os
import numpy as np
import torch
from hydra.utils import instantiate
import hydra
from flextok.utils.misc import detect_bf16_support, get_bf16_context
from torchvision.utils import save_image
import wandb


"""
Purpose
-------
Reconstruct ImageNet images from precomputed token IDs and save them on disk under:
        {output_path}/reconst_{k}/{WNID}/{original_filename}

Output range
------------
FlexTok decoder produces images in [-1, 1]. We clamp to [-1,1], map to [0,1], and save.

Default inference settings
--------------------------
We keep timesteps=20 and guidance_scale=7.5 as defaults per the request.
"""


@hydra.main(version_base=None, config_path="conf", config_name="prep_dataset_compression_reconstruction")
def main(cfg):
    device = torch.device(cfg.device)

    # Instantiate model
    model = instantiate(cfg.experiment.model).eval().to(device)

    # Config (paths and parameters coming from Hydra config)
    k_keep_list = cfg.experiment.k_keep_list                 # e.g., [1,2,4,...,256]
    base_out = cfg.experiment.output_path                    # parent output dir
    data_root = cfg.experiment.data_root                     # ImageNet train root with 1000 WNID subfolders
    batch_size = int(cfg.experiment.get("batch_size", 64))  # decode batch size
    start_class_idx = int(cfg.experiment.get("start_class_idx", 0))  # inclusive
    end_class_idx = int(cfg.experiment.get("end_class_idx", 1000))   # exclusive
    # Defaults requested
    timesteps = 20
    guidance_scale = 7.5

    # Tokens
    tokens_path = cfg.experiment.register_tokens_path
    assert os.path.isfile(tokens_path), f"register_tokens_path not found: {tokens_path}"
    npz = np.load(tokens_path, mmap_mode="r")
    if "token_ids" not in npz.files:
        raise KeyError(f"NPZ at {tokens_path} must contain 'token_ids'")
    token_ids = npz["token_ids"]  # memmap array [N, 256]

    # 1) Build sorted list of WNIDs (class folders)
    wnids = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    num_classes = len(wnids)
    if end_class_idx > num_classes:
        end_class_idx = num_classes
    if start_class_idx < 0 or start_class_idx >= end_class_idx:
        raise ValueError(f"Invalid class range: [{start_class_idx}, {end_class_idx}) with total classes={num_classes}")

    # 2) Compute global index offset for the first selected class by counting files in all prior classes.
    #    We do NOT store file paths; we only sum counts for classes [0 .. start_class_idx-1].
    def _count_files(cls_dir: str) -> int:
        # Count files (non-recursive) in a class directory
        return sum(1 for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f)))

    global_offset = 0
    for i in range(start_class_idx, 0, -1):
        # Walk backwards to reduce string concatenations? Not necessary; keep simple forward loop instead.
        pass
    # Simple forward accumulation
    global_offset = 0
    for i in range(0, start_class_idx):
        cls_dir = os.path.join(data_root, wnids[i])
        global_offset += _count_files(cls_dir)

    N_tokens = token_ids.shape[0]

    enable_bf16 = detect_bf16_support()
    print("BF16 enabled:", enable_bf16)

    # Initialize Weights & Biases
    wandb.init(
        project="imagenet_reconstruct_tokens",
        name=f"reconst_classes_{cfg.experiment.start_class_idx}_{cfg.experiment.end_class_idx}",
        config={
            "k_keep_list": list(map(int, k_keep_list)),
            "data_root": data_root,
            "output_path": base_out,
            "batch_size": batch_size,
            "start_class_idx": start_class_idx,
            "end_class_idx": end_class_idx
        },
        reinit=True,
    )

    # Pre-create top-level reconst_{k} directories
    for k in k_keep_list:
        os.makedirs(os.path.join(base_out, f"reconst_{k}"), exist_ok=True)

    total_saved = {int(k): 0 for k in k_keep_list}

    # 3) Iterate over selected classes and reconstruct/save their images in mini-batches.
    current_global = global_offset  # running global index into token_ids
    for ci in range(start_class_idx, end_class_idx):
        wnid = wnids[ci]
        cls_dir = os.path.join(data_root, wnid)

        # Files for this class (alphabetically sorted), used to derive original filenames
        names = sorted([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])
        num_files = len(names)

        if current_global + num_files > N_tokens:
            raise RuntimeError(
                f"Token array too short: need index up to {current_global+num_files-1}, have N_tokens={N_tokens}"
            )

        print(f"Processing class {ci}/{num_classes}: {wnid} with {num_files} files (global start={current_global})")

        # Process this class in mini-batches to keep memory manageable
        for base_idx in range(0, num_files, batch_size):
            B = min(batch_size, num_files - base_idx)
            local_idxs = list(range(base_idx, base_idx + B))
            global_idxs = [current_global + j for j in local_idxs]

            for k_keep in k_keep_list:
                tokens_list_filtered = []
                out_paths = []

                # Build tokens for this mini-batch and the corresponding output paths
                for li, gi in zip(local_idxs, global_idxs):
                    ids_np = token_ids[gi, :k_keep]
                    ids_t = torch.from_numpy(np.asarray(ids_np, dtype=np.int64)).unsqueeze(0).to(device)
                    tokens_list_filtered.append(ids_t)

                    basename = names[li]
                    out_dir = os.path.join(base_out, f"reconst_{k_keep}", wnid)
                    os.makedirs(out_dir, exist_ok=True)
                    out_paths.append(os.path.join(out_dir, basename))

                # Detokenize and save this mini-batch
                with get_bf16_context(enable_bf16):
                    reconstructed = model.detokenize(
                        tokens_list_filtered,
                        timesteps=timesteps,
                        guidance_scale=guidance_scale, verbose=False,
                    )  # [-1,1]
                for img_t, out_path in zip(reconstructed, out_paths):
                    save_image(img_t, out_path)
                    total_saved[int(k_keep)] += 1

        # Advance the global token offset by number of files in this class
        current_global += num_files

        print(f"Finished class {wnid}; cumulative saved per k: {total_saved}")
        # Log per-class completion to W&B (step by class index)
        wandb.log({
            "class/index": ci,
            "class/wnid": wnid,
            **{f"saved/reconst_{k}": total_saved[int(k)] for k in k_keep_list},
        }, step=ci)

    wandb.finish()

    print("Done. Final saved counts per k:", total_saved)

    
if __name__ == "__main__":
    main()