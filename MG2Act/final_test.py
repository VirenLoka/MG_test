import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import MG2ActDataset, collate_samples
from .model import MG2ActModel
from .result_record import evaluate_regression_metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_frozen_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    config = checkpoint["config"]

    model = MG2ActModel(
        device=device,
        embed_dim=config["embed_dim"],
        attn_heads=config["attn_heads"],
        decoder_layers=config["decoder_layers"],
        mlp_hidden=config["mlp_hidden"],
        dropout=config["dropout"],
        proj_method=config["proj_method"],
        gnn_type=config["gnn_type"],
        gnn_layers=config["gnn_layers"],
        gnn_hidden_dim=config["gnn_hidden_dim"],
        enable_fg_boost=config["enable_fg_boost"],
        fusion_method=config["fusion_method"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Final evaluation on the untouched test set"
    )

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("final_test_results.json"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    # 防止无意中反复覆盖最终测试结果
    if args.output.exists():
        raise FileExistsError(
            f"{args.output} already exists. "
            "The final test evaluation should not be repeatedly overwritten."
        )

    device = torch.device(args.device)

    test_dataset = MG2ActDataset(
        args.test_csv,
        col_activity="Score",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )

    model = load_frozen_model(
        args.checkpoint,
        device,
    )

    test_metrics = evaluate_regression_metrics(
        model,
        test_loader,
        device,
    )

    results = {
        "evaluation_stage": "final_test_only",
        "model_selection_criterion": "validation_loss",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "test_csv": str(args.test_csv),
        "test_csv_sha256": sha256_file(args.test_csv),
        "test_metrics": test_metrics,
    }

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""
python -m MG2Act.final_test \
  --checkpoint outputs/mg2act_best.pt \
  --test_csv MG_data/test.csv \
  --output outputs/final_test_results.json \
  --device cuda:0
"""
