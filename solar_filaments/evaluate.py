import argparse
import json
from pathlib import Path
from pycocotools import mask as mask_utils
from .data import FilamentDataset
from .metrics import official_pq, aggregate_pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-root", default="data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="runs/evaluation.json")
    args = parser.parse_args()
    saved = json.loads(Path(args.predictions).read_text())
    predictions = {image["file_name"]: [instance["segmentation"] for instance in image["instances"]]
                   for image in saved["predictions"]}
    dataset = FilamentDataset(args.data_root, "val", args.fold, args.folds, args.seed)
    missing = {record["file_name"] for record in dataset.records} - predictions.keys()
    if missing:
        raise ValueError(f"Validation predictions missing for {len(missing)} physical images")
    records = []
    for record in dataset.records:
        ground_truth = []
        for annotation in dataset.annotations[str(record["id"])]:
            rles = mask_utils.frPyObjects(annotation["segmentation"], record["height"], record["width"])
            ground_truth.append(mask_utils.merge(rles) if isinstance(rles, list) else rles)
        records.append({"record_id": str(record["id"]), "file_name": record["file_name"],
                        **official_pq(predictions[record["file_name"]], ground_truth)})
    result = {"task_trained": saved.get("task_trained", False), **aggregate_pq(records), "records": records}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
