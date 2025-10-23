import os
import numpy as np
import torch
from hydra.utils import instantiate
import hydra
from flextok.utils.misc import detect_bf16_support, get_bf16_context
from torchvision.utils import save_image
import wandb


"""
This script reconstructs images from token IDs stored in a .npz file using FlexTok
for different token counts (k_keep). It now accepts an explicit list of class
folder names (WNIDs) via cfg.experiment.class_folders, matching the submit script.
"""


@hydra.main(version_base=None, config_path="conf", config_name="prep_dataset_compression_reconstruction")
def main(cfg):
    device = torch.device(cfg.device)

    # Instantiate FlexTok model (decoder)
    model = instantiate(cfg.experiment.model).eval().to(device)

    # Config
    k_keep_list = cfg.experiment.k_keep_list                 # e.g., [1,2,4,...,256]
    base_out = cfg.experiment.output_path                    # parent output dir
    data_root = cfg.experiment.data_root                     # ImageNet train root (WNID folders)
    batch_size = int(cfg.experiment.get("batch_size", 64))  # decode batch size
    class_folders = list(cfg.experiment.get("class_folders", []))
    if not class_folders:
        raise ValueError("experiment.class_folders must be provided (list of WNIDs)")

    # Inference defaults
    timesteps = 20
    guidance_scale = 7.5

    # Load precomputed token IDs
    tokens_path = cfg.experiment.register_tokens_path
    assert os.path.isfile(tokens_path), f"register_tokens_path not found: {tokens_path}"
    npz = np.load(tokens_path, mmap_mode="r")
    if "token_ids" not in npz.files:
        raise KeyError(f"NPZ at {tokens_path} must contain 'token_ids'")
    token_ids = npz["token_ids"]  # [N, 256]
    N_tokens = int(token_ids.shape[0])

    # Build sorted global WNID list from data_root and compute offsets by counting files
    all_wnids = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

    def _count_files(cls_dir: str) -> int:
        return sum(1 for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f)))

    counts = [_count_files(os.path.join(data_root, wn)) for wn in all_wnids]
    offsets = [0]
    for c in counts[:-1]:
        offsets.append(offsets[-1] + c)
    wnid_to_offset = {wn: off for wn, off in zip(all_wnids, offsets)}
    wnid_to_count = {wn: c for wn, c in zip(all_wnids, counts)}

    # Validate requested class folders
    for wn in class_folders:
        if wn not in wnid_to_offset:
            raise ValueError(f"Requested class folder '{wn}' not found under data_root: {data_root}")

    enable_bf16 = detect_bf16_support()
    print("BF16 enabled:", enable_bf16)

    # W&B init
    wandb.init(
        project="imagenet_reconstruct_tokens",
        name=f"reconst_folders_{len(class_folders)}",
        config={
            "k_keep_list": list(map(int, k_keep_list)),
            "data_root": data_root,
            "output_path": base_out,
            "batch_size": batch_size,
            "class_folders": class_folders,
        },
        reinit=True,
    )

    # Pre-create top-level reconst_{k} directories
    for k in k_keep_list:
        os.makedirs(os.path.join(base_out, f"reconst_{k}"), exist_ok=True)

    total_saved = {int(k): 0 for k in k_keep_list}

    # Iterate over requested classes
    for wnid in class_folders:
        cls_dir = os.path.join(data_root, wnid)
        names = sorted([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])
        num_files = len(names)
        global_start = wnid_to_offset[wnid]
        if global_start + num_files > N_tokens:
            raise RuntimeError(
                f"Token array too short: need up to index {global_start+num_files-1}, have N_tokens={N_tokens}"
            )

        print(f"Processing class {wnid} with {num_files} files (global start={global_start})")

        # Process this class in mini-batches
        for base_idx in range(0, num_files, batch_size):
            B = min(batch_size, num_files - base_idx)
            local_idxs = list(range(base_idx, base_idx + B))
            global_idxs = [global_start + j for j in local_idxs]

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
                        guidance_scale=guidance_scale,
                        verbose=False,
                    )  # [-1,1]
                for img_t, out_path in zip(reconstructed, out_paths):
                    save_image(((img_t.clamp(-1, 1) + 1) / 2), out_path)
                    total_saved[int(k_keep)] += 1

        print(f"Finished class {wnid}; cumulative saved per k: {total_saved}")
        wandb.log({
            "class/wnid": wnid,
            **{f"saved/reconst_{k}": total_saved[int(k)] for k in k_keep_list},
        })

    wandb.finish()
    print("Done. Final saved counts per k:", total_saved)

    
if __name__ == "__main__":
    main()