from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmt.config import load_config, save_config
from tmt.jobs import FINAL_STATUSES, list_jobs, request_stop, start_job, tail_log
from tmt.training import checkpoint_metadata


st.set_page_config(page_title="TMT Console", page_icon="T", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f7f8f8; color: #172026; }
    [data-testid="stHeader"] { background: rgba(247, 248, 248, 0.94); }
    [data-testid="stSidebar"] { background: #eef1f1; border-right: 1px solid #d8dede; }
    .block-container { max-width: 1440px; padding-top: 2rem; }
    h1, h2, h3 { letter-spacing: 0 !important; }
    h1 { font-size: 1.7rem !important; }
    h2 { font-size: 1.2rem !important; }
    [data-testid="stMetric"] { background: transparent; border-left: 2px solid #16867a; padding-left: 0.8rem; }
    [data-testid="stMetricValue"] { font-size: 1.35rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid #d8dede; }
    .stTabs [data-baseweb="tab"] { height: 2.5rem; border-radius: 0; }
    .stTabs [aria-selected="true"] { border-bottom: 2px solid #16867a; }
    </style>
    """,
    unsafe_allow_html=True,
)


def script_command(script: str, *arguments: object) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / script), *(str(value) for value in arguments)]


@st.cache_data(ttl=2)
def find_checkpoints() -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    for path in sorted((ROOT / "checkpoints").glob("**/*.safetensors"), reverse=True):
        try:
            metadata = checkpoint_metadata(path)
        except (OSError, ValueError):
            continue
        checkpoints.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(ROOT)),
                "kind": metadata.get("kind", "unknown"),
                "step": int(metadata["step"]) if metadata.get("step") else None,
                "run_id": metadata.get("run_id") or path.parent.name,
                "size_mb": round(path.stat().st_size / 1_000_000, 1),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return checkpoints


def checkpoint_select(label: str, *, kind: str | None = None, key: str) -> dict[str, Any] | None:
    checkpoints = [item for item in find_checkpoints() if kind is None or item["kind"] == kind]
    options = ["None", *(item["relative_path"] for item in checkpoints)]
    selected = st.selectbox(label, options, key=key)
    if selected == "None":
        return None
    return next(item for item in checkpoints if item["relative_path"] == selected)


def device_options() -> list[str]:
    options = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    return [*options, "cpu"]


def valid_run_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value)) and value not in {".", ".."}


def start_and_report(
    kind: str,
    command: list[str],
    parameters: dict[str, Any],
    *,
    environment: dict[str, str] | None = None,
) -> None:
    job = start_job(kind, command, parameters, environment=environment)
    st.success(f"Job queued: {job['job_id']}")


jobs = list_jobs()
checkpoints = find_checkpoints()
active_jobs = sum(job["status"] not in FINAL_STATUSES for job in jobs)

with st.sidebar:
    st.subheader("Runtime")
    st.metric("CUDA", "Available" if torch.cuda.is_available() else "Unavailable")
    st.metric("GPUs", torch.cuda.device_count())
    st.metric("Active jobs", active_jobs)
    st.metric("Checkpoints", len(checkpoints))
    if torch.cuda.is_available():
        st.caption(torch.cuda.get_device_name(0))

st.title("TMT Console")
st.caption("Training, inference, evaluation, datasets, and run state")

train_tab, generate_tab, evaluate_tab, dataset_tab, jobs_tab, checkpoint_tab = st.tabs(
    ["Train", "Generate", "Evaluate", "Datasets", "Jobs", "Checkpoints"]
)

config_paths = sorted((ROOT / "configs").glob("*.yaml"))
config_labels = [str(path.relative_to(ROOT)) for path in config_paths]
default_config_index = next(
    (index for index, label in enumerate(config_labels) if label.endswith("tmt_5m.yaml")),
    0,
)

with train_tab:
    left, right = st.columns([1, 1], gap="large")
    with left:
        config_label = st.selectbox("Base config", config_labels, key="train_config")
        resume_checkpoint = checkpoint_select("Resume checkpoint", kind="training", key="train_resume")
    selected_config = load_config(ROOT / config_label)
    default_run_id = f"{selected_config.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with right:
        st.metric("Parameters", f"{selected_config.parameter_count_formula:,}")
        if resume_checkpoint:
            st.metric("Resume step", f"{resume_checkpoint['step']:,}")

    with st.form("train_form"):
        run_id = st.text_input(
            "Run ID",
            value=st.session_state.get("suggested_run_id", default_run_id),
            help="Leave blank while resuming to continue in the checkpoint's run directory.",
        )
        data_pattern = st.text_input("Data glob", value=selected_config.stream.data_glob)
        row = st.columns(4)
        target_step = row[0].number_input("Target step", min_value=1, value=10_000, step=500)
        device = row[1].selectbox("Device", device_options(), key="train_device")
        checkpoint_interval = row[2].number_input(
            "Checkpoint interval", min_value=1, value=selected_config.checkpoint.interval, step=100
        )
        batch_size = row[3].number_input(
            "Batch size", min_value=1, value=selected_config.stream.batch_size, step=1
        )

        row = st.columns(4)
        dim = row[0].number_input("Dimension", min_value=8, value=selected_config.dim, step=8)
        layers = row[1].number_input("Layers", min_value=1, value=selected_config.num_layers, step=1)
        learning_rate = row[2].number_input(
            "Learning rate",
            min_value=0.000001,
            max_value=1.0,
            value=selected_config.optimizer.learning_rate,
            format="%.6f",
        )
        reset_policy = row[3].selectbox(
            "Reset policy",
            ["document", "fixed_byte_interval", "never"],
            index=["document", "fixed_byte_interval", "never"].index(selected_config.stream.reset_policy),
        )
        autocast = st.toggle("BF16 autocast", value=selected_config.precision.autocast)
        submitted = st.form_submit_button(
            "Start training", type="primary", icon=":material/play_arrow:", width="content"
        )

    if submitted:
        effective_run_id = run_id.strip()
        if not effective_run_id and resume_checkpoint:
            effective_run_id = str(resume_checkpoint["run_id"])
        if not valid_run_id(effective_run_id):
            st.error("Run ID must contain only letters, numbers, dots, underscores, or hyphens.")
        else:
            config = load_config(ROOT / config_label)
            config.dim = int(dim)
            config.num_layers = int(layers)
            config.optimizer.learning_rate = float(learning_rate)
            config.stream.data_glob = data_pattern
            config.stream.batch_size = int(batch_size)
            config.stream.reset_policy = reset_policy
            config.precision.autocast = autocast
            config.checkpoint.interval = int(checkpoint_interval)
            config.checkpoint.resume = None
            generated_config = ROOT / "runs" / effective_run_id / "config.yaml"
            save_config(config, generated_config)
            command = script_command(
                "train.py",
                "--config",
                generated_config,
                "--data",
                data_pattern,
                "--device",
                device,
                "--max-steps",
                int(target_step),
                "--run-id",
                effective_run_id,
            )
            if resume_checkpoint:
                command.extend(["--resume", resume_checkpoint["path"]])
            start_and_report(
                "train",
                command,
                {
                    "run_id": effective_run_id,
                    "config": config_label,
                    "data": data_pattern,
                    "target_step": int(target_step),
                    "device": device,
                    "resume": resume_checkpoint["relative_path"] if resume_checkpoint else None,
                },
            )

with generate_tab:
    generation_config = st.selectbox(
        "Config", config_labels, index=default_config_index, key="generation_config"
    )
    generation_checkpoint = checkpoint_select("Checkpoint", key="generation_checkpoint")
    with st.form("generation_form"):
        prompt = st.text_area("Prompt", height=140)
        row = st.columns(4)
        max_new_bytes = row[0].number_input("Maximum new bytes", min_value=1, value=256, step=32)
        threshold = row[1].slider("Stop threshold", min_value=0.0, max_value=1.0, value=0.35)
        generation_device = row[2].selectbox("Device", device_options(), key="generation_device")
        online = row[3].toggle("Online updates", value=False)
        submitted = st.form_submit_button("Generate", type="primary", icon=":material/auto_awesome:")
    if submitted:
        command = script_command(
            "generate.py",
            "--config",
            ROOT / generation_config,
            "--prompt",
            prompt,
            "--max-new-bytes",
            int(max_new_bytes),
            "--threshold",
            threshold,
            "--device",
            generation_device,
        )
        if generation_checkpoint:
            command.extend(["--checkpoint", generation_checkpoint["path"]])
        if online:
            command.append("--online")
        start_and_report(
            "generate",
            command,
            {"checkpoint": generation_checkpoint["relative_path"] if generation_checkpoint else None},
        )

with evaluate_tab:
    evaluation_config = st.selectbox(
        "Config", config_labels, index=default_config_index, key="evaluation_config"
    )
    evaluation_checkpoint = checkpoint_select("Checkpoint", key="evaluation_checkpoint")
    with st.form("evaluation_form"):
        evaluation_data = st.text_input("Evaluation file", value="")
        row = st.columns(2)
        max_bytes = row[0].number_input("Maximum bytes", min_value=1, value=100_000, step=10_000)
        evaluation_device = row[1].selectbox("Device", device_options(), key="evaluation_device")
        submitted = st.form_submit_button("Run evaluation", type="primary", icon=":material/analytics:")
    if submitted:
        command = script_command(
            "evaluate.py",
            "--config",
            ROOT / evaluation_config,
            "--device",
            evaluation_device,
            "--max-bytes",
            int(max_bytes),
        )
        if evaluation_checkpoint:
            command.extend(["--checkpoint", evaluation_checkpoint["path"]])
        if evaluation_data:
            command.extend(["--data", evaluation_data])
        start_and_report(
            "evaluate",
            command,
            {
                "checkpoint": evaluation_checkpoint["relative_path"] if evaluation_checkpoint else None,
                "data": evaluation_data or None,
            },
        )

with dataset_tab:
    with st.form("dataset_form"):
        dataset_id = st.text_input("Dataset ID or URL", value="Open-Orca/OpenOrca")
        output_dir = st.text_input("Output directory", value="")
        row = st.columns(3)
        revision = row[0].text_input("Revision", value="main")
        include = row[1].text_input("Include patterns", value="*.parquet")
        exclude = row[2].text_input("Exclude patterns", value="")
        hf_token = st.text_input(
            "Hugging Face token",
            value="",
            type="password",
            help="Optional. Passed as HF_TOKEN for this job and never written to job metadata or logs.",
        )
        dry_run = st.toggle("Dry run", value=True)
        submitted = st.form_submit_button("Start download", type="primary", icon=":material/download:")
    if submitted:
        command = script_command("download_dataset.py", dataset_id, "--revision", revision)
        if output_dir:
            command.extend(["--output", output_dir])
        for pattern in [item.strip() for item in include.split(",") if item.strip()]:
            command.extend(["--include", pattern])
        for pattern in [item.strip() for item in exclude.split(",") if item.strip()]:
            command.extend(["--exclude", pattern])
        if dry_run:
            command.append("--dry-run")
        start_and_report(
            "dataset",
            command,
            {"dataset": dataset_id, "revision": revision, "dry_run": dry_run},
            environment={"HF_TOKEN": hf_token.strip()} if hf_token.strip() else None,
        )

with jobs_tab:

    @st.fragment(run_every=2)
    def render_jobs() -> None:
        current_jobs = list_jobs()
        row = st.columns(4)
        row[0].metric("Total", len(current_jobs))
        row[1].metric("Running", sum(job["status"] == "running" for job in current_jobs))
        row[2].metric("Queued", sum(job["status"] == "queued" for job in current_jobs))
        row[3].metric("Failed", sum(job["status"] == "failed" for job in current_jobs))
        if not current_jobs:
            st.info("No jobs recorded.")
            return
        st.dataframe(
            [
                {
                    "job": job["job_id"],
                    "type": job["kind"],
                    "status": job["status"],
                    "created": job["created_at"],
                    "return code": job["return_code"],
                }
                for job in current_jobs
            ],
            width="stretch",
            hide_index=True,
        )
        selected_id = st.selectbox("Job log", [job["job_id"] for job in current_jobs])
        selected_job = next(job for job in current_jobs if job["job_id"] == selected_id)
        controls = st.columns([1, 5])
        if controls[0].button(
            "Stop",
            icon=":material/stop:",
            disabled=selected_job["status"] in FINAL_STATUSES,
            key=f"stop-{selected_id}",
        ):
            request_stop(selected_id)
            st.rerun(scope="fragment")
        log = tail_log(selected_job)
        st.code(log or "Waiting for output...", language="text")

    render_jobs()

with checkpoint_tab:
    current_checkpoints = find_checkpoints()
    if current_checkpoints:
        st.dataframe(
            [
                {
                    "run": item["run_id"],
                    "kind": item["kind"],
                    "step": item["step"],
                    "size MB": item["size_mb"],
                    "modified": item["modified"],
                    "path": item["relative_path"],
                }
                for item in current_checkpoints
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No safetensors checkpoints found.")
