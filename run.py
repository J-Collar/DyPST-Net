import argparse
import os
import pickle
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

import yaml

from exp.exp_sst import run_experiment


def _normalize_frequency(value: str) -> str:
    freq = str(value or "").strip().lower()
    if freq in {"day", "daily"}:
        return "daily"
    if freq in {"week", "weekly"}:
        return "weekly"
    return freq


def _infer_frequency_from_data_path(data_path: str) -> str:
    norm_parts = [p.lower() for p in os.path.normpath(data_path).split(os.sep) if p]
    for candidate in ("daily", "weekly"):
        if candidate in norm_parts:
            return candidate

    split_info_path = os.path.join(data_path, "split_info.pkl")
    if os.path.exists(split_info_path):
        try:
            with open(split_info_path, "rb") as f:
                meta = pickle.load(f)
            if isinstance(meta, dict):
                return _normalize_frequency(meta.get("frequency", ""))
        except Exception:
            return ""
    return ""


def _make_run_tag(region: str, frequency: str) -> str:
    return f"{region}_{frequency}" if frequency and frequency != "unknown" else region


def load_config(config_path: str, region_override: str = "", frequency_override: str = "", data_path_override: str = ""):
    project_root = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isabs(config_path):
        config_path = os.path.normpath(os.path.join(project_root, config_path))

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if data_path_override:
        cfg.setdefault("data", {})["path"] = data_path_override

    data_path = cfg["data"]["path"]
    if not os.path.isabs(data_path):
        cfg["data"]["path"] = os.path.normpath(os.path.join(project_root, data_path))

    if region_override:
        cfg.setdefault("data", {})["region"] = region_override
    if frequency_override:
        cfg.setdefault("data", {})["frequency"] = frequency_override

    region = str(cfg.get("data", {}).get("region", "")).strip().lower()
    if not region:
        region = os.path.basename(os.path.normpath(cfg["data"]["path"])).lower()
    cfg.setdefault("data", {})["region"] = region

    frequency = _normalize_frequency(cfg.get("data", {}).get("frequency", ""))
    if not frequency:
        frequency = _infer_frequency_from_data_path(cfg["data"]["path"])
    if not frequency:
        frequency = "unknown"
    cfg["data"]["frequency"] = frequency
    run_tag = _make_run_tag(region, frequency)
    cfg["data"]["run_tag"] = run_tag

    ckpt_dir = cfg["exp"].get("ckpt_dir", "./checkpoints")
    if frequency != "unknown" and "{frequency}" not in ckpt_dir and "{run_tag}" not in ckpt_dir:
        ckpt_dir = os.path.join(ckpt_dir, "{frequency}")
    ckpt_dir = ckpt_dir.format(region=region, frequency=frequency, run_tag=run_tag)
    if not os.path.isabs(ckpt_dir):
        cfg["exp"]["ckpt_dir"] = os.path.normpath(os.path.join(project_root, ckpt_dir))
    else:
        cfg["exp"]["ckpt_dir"] = ckpt_dir
    cfg["exp"]["run_tag"] = run_tag

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="SST Tri-Branch Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument("--region", type=str, default="", help="Override data.region, e.g. ecs or scs")
    parser.add_argument("--frequency", type=str, choices=["daily", "weekly"], default="", help="Override data.frequency")
    parser.add_argument("--data-path", type=str, default="", help="Override data.path")
    return parser.parse_args()


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _print_config_snapshot(config, config_path):
    print("=" * 80)
    print("Experiment Configuration Snapshot")
    print("=" * 80)
    print(f"Config path: {config_path}")
    print(yaml.safe_dump(config, sort_keys=False, allow_unicode=False))
    print("=" * 80)


def _format_final_log_name(run_tag: str, final_mae: float, end_time: datetime) -> str:
    return f"{run_tag}_{final_mae:.6f}_{end_time.strftime('%Y%m%d_%H%M%S')}.log"


def _step_label(frequency: str) -> str:
    if frequency == "daily":
        return "Day"
    if frequency == "weekly":
        return "Week"
    return "Step"


def run_with_logging(config, config_path):
    region = str(config.get("data", {}).get("region", "unknown")).strip().lower() or "unknown"
    frequency = str(config.get("data", {}).get("frequency", "unknown")).strip().lower() or "unknown"
    run_tag = str(config.get("data", {}).get("run_tag", _make_run_tag(region, frequency))).strip().lower()
    step_label = _step_label(frequency)
    ckpt_dir = config["exp"].get("ckpt_dir", "./checkpoints")
    log_dir = os.path.join(ckpt_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    start_time = datetime.now()
    tmp_log_path = os.path.join(log_dir, f"{run_tag}_running_{start_time.strftime('%Y%m%d_%H%M%S')}.log")

    artifacts = None

    with open(tmp_log_path, "w", encoding="utf-8") as log_f:
        tee_out = TeeStream(sys.stdout, log_f)
        tee_err = TeeStream(sys.stderr, log_f)

        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"Experiment start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Experiment identity: region={region}, frequency={frequency}, run_tag={run_tag}")
            _print_config_snapshot(config, config_path)

            try:
                artifacts = run_experiment(config)

                print("Training completed.")
                print(f"Best checkpoint: {artifacts.best_ckpt}")

            except Exception as exc:
                print("\nExperiment failed with exception:")
                print(repr(exc))
                print(traceback.format_exc())
                raise
            finally:
                end_time = datetime.now()
                print(f"Experiment end time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Duration: {end_time - start_time}")

    end_time = datetime.now()
    if artifacts is not None:
        final_name = _format_final_log_name(run_tag, artifacts.test_mae_celsius, end_time)
    else:
        final_name = f"{run_tag}_FAILED_{end_time.strftime('%Y%m%d_%H%M%S')}.log"
    final_log_path = os.path.join(log_dir, final_name)
    os.replace(tmp_log_path, final_log_path)
    return artifacts, final_log_path


if __name__ == "__main__":
    args = parse_args()
    config = load_config(
        args.config,
        region_override=args.region,
        frequency_override=args.frequency,
        data_path_override=args.data_path,
    )
    _, log_path = run_with_logging(config, args.config)
    print(f"Experiment log saved to: {log_path}")
