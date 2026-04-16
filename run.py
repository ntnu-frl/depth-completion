import argparse
import yaml

from pathlib import Path

from depth_completion import DepthCompletionModel
from evaluation.loader import DatasetLoader

def format_results(results):
    def fmt(value):
        text = f"{value:.3f}"
        text = text.rstrip("0").rstrip(".")
        return text

    parts = []
    for dataset_name, metrics in results.items():
        parts.append(f"{dataset_name}/mean/mae: {fmt(metrics['mae'])}")
        parts.append(f"{dataset_name}/mean/rmse: {fmt(metrics['rmse'])}")
    return " ".join(parts)


def evaluate_dataset(model, dataset_name, config, dataset_root, dataset_loader_cls):
    loader = dataset_loader_cls(dataset_root / "evaluation" / dataset_name, config)
    mae_values = []
    rmse_values = []

    for sample in loader:
        prediction = model.predict(sample["image"], sample["prior_points"], sample["intrinsics"])
        depth_gt = sample["depth"]
        mask = (depth_gt > config["depth_min"]) & (depth_gt <= config["depth_max"])
        if mask.sum() == 0:
            continue
        diff = prediction[mask] - depth_gt[mask]
        mae_values.append(float(abs(diff).mean()))
        rmse_values.append(float((diff ** 2).mean() ** 0.5))

    if not mae_values:
        raise RuntimeError(f"No valid samples found for {dataset_name}")

    return {
        "mae": sum(mae_values) / len(mae_values),
        "rmse": sum(rmse_values) / len(rmse_values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("r") as handle:
        config = yaml.safe_load(handle)

    model = DepthCompletionModel(config_path=args.config)

    results = {}
    for public_name in config["datasets"]:
        results[public_name] = evaluate_dataset(
            model,
            public_name,
            config,
            args.dataset_root,
            DatasetLoader,
        )

    print(format_results(results))


if __name__ == "__main__":
    main()
