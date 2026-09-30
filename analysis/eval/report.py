"""Donanım gerektirmeyen, deterministik metrik doğrulama çıktısı."""

import json
from dataclasses import asdict

from eval.metrics import reprojection_error


def main():
    result = reprojection_error([[0, 0], [0, 0]], [[3, 4], [0, 0]])
    print(json.dumps({
        "experiment": "synthetic_metric_sanity_check",
        "description": "İki noktanın bilinen hataları: 5 px ve 0 px.",
        "physical_validation": False,
        "reprojection_error": asdict(result),
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
