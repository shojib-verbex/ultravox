"""
Evaluation script for Qwen3-Omni models using vLLM backend.

This script mirrors the structure of eval.py but uses Qwen3OmniVLLMInference
for evaluating Qwen3-Omni models on speech/audio benchmarks.

Usage:
    # Run evaluation with vLLM server
    python -m ultravox.evaluation.eval_qwen3 \
        --config_path configs/eval_config_qwen3_omni.yaml \
        --vllm_url http://localhost:7000

    # Quick test with limited samples
    python -m ultravox.evaluation.eval_qwen3 \
        --model "Qwen/Qwen3-Omni-30B-A3B-Instruct" \
        --eval_sets '[{"name": "slurp-intent"}]' \
        --eval_dataset_args.max_samples 10

Prerequisites:
    Start vLLM server first:
        docker compose up -d vllm-server
"""
import dataclasses
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import simple_parsing
import wandb
from tqdm import tqdm

from ultravox import data
from ultravox.evaluation import eval_metrics
from ultravox.evaluation import eval_types
from ultravox.inference import base as infer_base
from ultravox.inference import qwen3_omni_vllm_infer

logging.basicConfig(level=logging.INFO)


@dataclasses.dataclass
class Qwen3EvalConfig:
    """Configuration for Qwen3-Omni evaluation using vLLM."""

    model: str
    eval_sets: List[Dict[str, Any]] = simple_parsing.list_field()
    eval_dataset_args: data.EvalDatasetArgs = simple_parsing.field(
        default_factory=data.EvalDatasetArgs
    )

    # vLLM server settings
    vllm_url: str = "http://localhost:7000"

    # Evaluation parameters
    eval_batch_size: int = 4  # Batch size for concurrent requests
    eval_max_tokens: int = 512
    eval_temperature: float = 0.0

    # Output settings
    exp_name: Optional[str] = None
    output_dir: Optional[Path] = None
    report_logs_to: List[str] = simple_parsing.list_field()

    def get_eval_sets(self) -> List[data.DatasetOptions]:
        return [data.DatasetOptions(**ds) for ds in self.eval_sets]

    def __post_init__(self):
        if self.exp_name is None:
            self.exp_name = datetime.datetime.now().strftime(
                "qwen3-eval-vllm--%Y-%m-%d--%H-%M-%S"
            )
        if self.output_dir is None:
            self.output_dir = Path("runs") / self.exp_name


def infer_dataset(
    inference: infer_base.VoiceInference,
    dataset: data.SizedIterableDataset,
    batch_size: int = 1,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    output_file: Optional[str] = None,
    num_prefetch_batches: int = 2,
) -> List[eval_types.Sample]:
    """
    Run inference on a dataset with incremental result saving and prefetching.

    Uses background threads to prefetch batches while GPU inference is running,
    minimizing GPU idle time between batches.

    Args:
        inference: VoiceInference instance
        dataset: Dataset to evaluate
        batch_size: Batch size for inference
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        output_file: Optional path to save results incrementally (JSONL format)
        num_prefetch_batches: Number of batches to prefetch in background (default: 2)

    Returns:
        List of Sample objects with predictions
    """
    import queue
    import sys
    import threading

    results = []
    total_samples = len(dataset)

    # Progress bar
    progress_bar = tqdm(
        total=total_samples,
        desc="Evaluating",
        unit="sample",
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.5,
    )

    # Open output file for incremental writing (JSONL format)
    output_handle = None
    if output_file:
        output_handle = open(output_file, "w")
        logging.info(f"Writing results incrementally to {output_file}")

    def write_results(batch_results: List[eval_types.Sample]):
        """Write batch results to file incrementally."""
        if output_handle:
            for result in batch_results:
                output_handle.write(
                    json.dumps(result.to_dict(), ensure_ascii=False) + "\n"
                )
            output_handle.flush()

    # Prefetch queue: holds prepared batches ready for inference
    prefetch_queue: queue.Queue = queue.Queue(maxsize=num_prefetch_batches)
    loading_done = threading.Event()
    loading_error: List[Exception] = []

    def batch_loader():
        """Background thread that loads and prepares batches."""
        try:
            batch = []
            for idx, sample in enumerate(dataset):
                # Pop the expected answer (assistant message)
                assistant_message = sample.messages.pop()
                if assistant_message["role"] != "assistant":
                    raise ValueError(
                        f"Expected assistant message but got: role={assistant_message['role']}"
                    )
                reference = assistant_message["content"]
                batch.append((idx, sample, reference))

                if len(batch) >= batch_size:
                    # Put batch in queue (blocks if queue is full)
                    prefetch_queue.put(batch)
                    batch = []

            # Put remaining samples
            if batch:
                prefetch_queue.put(batch)
        except Exception as e:
            loading_error.append(e)
        finally:
            loading_done.set()

    # Start background loader thread
    loader_thread = threading.Thread(target=batch_loader, daemon=True)
    loader_thread.start()

    try:
        while True:
            # Check for loading errors
            if loading_error:
                raise loading_error[0]

            # Try to get next batch
            try:
                batch = prefetch_queue.get(timeout=0.1)
            except queue.Empty:
                # Check if loading is done and queue is empty
                if loading_done.is_set() and prefetch_queue.empty():
                    break
                continue

            # Process batch
            batch_samples = [s for _, s, _ in batch]
            outputs = inference.infer_batch(batch_samples, max_tokens, temperature)

            batch_results = []
            for (index, sample, ref), output in zip(batch, outputs):
                result = eval_types.Sample(
                    index=index,
                    question=sample.messages[-1]["content"] if sample.messages else "",
                    transcript=sample.audio_transcript or "",
                    expected_answer=ref,
                    generated_answer=output.text,
                    extra_kwargs=sample.extra_kwargs,
                )
                results.append(result)
                batch_results.append(result)

            # Write batch results incrementally
            write_results(batch_results)

            # Update progress
            progress_bar.update(len(batch))
            sys.stdout.flush()

    finally:
        if output_handle:
            output_handle.close()
        # Wait for loader thread to finish
        loader_thread.join(timeout=1.0)

    progress_bar.close()
    results.sort(key=lambda x: x.index)
    return results


def eval_datasets(
    inference: infer_base.VoiceInference,
    dataset_options: List[data.DatasetOptions],
    dataset_args: data.EvalDatasetArgs,
    batch_size: int,
    max_tokens: Optional[int],
    temperature: float,
    output_dir: Optional[Path],
) -> Tuple[List[Tuple[str, str, float]], List[str]]:
    """
    Evaluate model on multiple datasets.

    Args:
        inference: VoiceInference instance
        dataset_options: List of dataset configurations
        dataset_args: Dataset arguments (max_samples, etc.)
        batch_size: Batch size for inference
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        output_dir: Directory to save results

    Returns:
        Tuple of (metrics list, output file paths)
    """
    metrics = []
    output_files = []

    for dataset_opt in dataset_options:
        logging.info(f"Evaluating on dataset: {dataset_opt.name}")

        dataset: Union[data.GenericDataset, data.Range] = data.create_dataset(
            dataset_opt.name, dataset_args, verbose=True
        )

        if dataset_args.max_samples != -1:
            dataset = data.Range(dataset, dataset_args.max_samples)

        # Determine output file path for incremental writing
        output_file = None
        if output_dir:
            output_file = os.path.join(output_dir, f"{dataset.name}.jsonl")

        results = infer_dataset(
            inference,
            dataset,
            batch_size=batch_size,
            max_tokens=max_tokens,
            temperature=temperature,
            output_file=output_file,
        )

        # Compute metrics if specified
        logging.info("Inference complete. Computing metrics...")
        dataset_config = dataset.get_config()
        aug_name = "null"

        if dataset_config.eval_config:
            eval_result: eval_types.Result = eval_metrics.evaluate_answers(
                results, dataset_config.eval_config
            )

            logging.info(
                f"Eval: {dataset.name}, "
                f"{dataset_config.eval_config.metric}: {eval_result.score:.2f}"
            )

            metrics.append(
                (
                    f"{dataset.name}.{dataset_config.eval_config.metric}",
                    aug_name,
                    eval_result.score,
                )
            )

            # Compute and report breakdown scores if configured
            if dataset_config.eval_config.breakdown_fields:
                for field in dataset_config.eval_config.breakdown_fields:
                    breakdown = eval_metrics.aggregate_scores_by_field(results, field)
                    for group_key, (score, count) in breakdown.items():
                        logging.info(
                            f"  Breakdown [{field}={group_key}]: {score:.4f} ({count} samples)"
                        )
                        metrics.append(
                            (
                                f"{dataset.name}.{dataset_config.eval_config.metric}.{group_key}",
                                aug_name,
                                score,
                            )
                        )

            if wandb.run:
                wandb.run.log(
                    {
                        "eval_results": wandb.Table(
                            columns=["metric", "augmentation", "score"],
                            data=metrics,
                        )
                    }
                )

        # Save final results with scores (JSON format)
        if output_dir:
            final_output_file = os.path.join(output_dir, f"{dataset.name}.json")
            with open(final_output_file, "w") as f:
                results_json = [result.to_dict() for result in results]
                json.dump(results_json, f, ensure_ascii=False, indent=2)
            logging.info(f"Final results with scores saved to {final_output_file}")
            output_files.append(final_output_file)

            if wandb.run:
                wandb.run.save(final_output_file)

    return metrics, output_files


def print_results(metrics: List[Tuple[str, str, float]], output_files: List[str]):
    """Print evaluation results summary."""
    print("\n" + "=" * 60)
    print("Qwen3-Omni Evaluation Results")
    print("=" * 60 + "\n")

    print("Scores:")
    for metric_name, augmentation_name, score in metrics:
        print(f"  {metric_name}: {score:.4f}")

    if output_files:
        print("\nOutput Files:")
        for output_file in output_files:
            print(f"  {output_file}")


def main(override_sys_args: Optional[List[str]] = None):
    """Main entry point for Qwen3-Omni evaluation."""
    config = simple_parsing.parse(
        Qwen3EvalConfig, add_config_path_arg=True, args=override_sys_args
    )

    logging.info("Starting Qwen3-Omni evaluation (vLLM backend)")
    logging.info(f"Model: {config.model}")
    logging.info(f"vLLM server URL: {config.vllm_url}")
    logging.info(f"Eval sets: {[ds['name'] for ds in config.eval_sets]}")

    # Create output directory
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if requested
    if "wandb" in config.report_logs_to:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "ultravox"),
            config=dataclasses.asdict(config),
            name=config.exp_name,
            dir="runs",
            save_code=True,
        )

    # Initialize Qwen3-Omni vLLM inference
    inference = qwen3_omni_vllm_infer.Qwen3OmniVLLMInference(
        model_path=config.model,
        vllm_url=config.vllm_url,
    )

    # Run evaluation
    metrics, output_files = eval_datasets(
        inference=inference,
        dataset_options=config.get_eval_sets(),
        dataset_args=config.eval_dataset_args,
        batch_size=config.eval_batch_size,
        max_tokens=config.eval_max_tokens,
        temperature=config.eval_temperature,
        output_dir=config.output_dir,
    )

    # Print results
    print_results(metrics, output_files)

    # Cleanup
    if wandb.run:
        wandb.run.finish()


if __name__ == "__main__":
    main()
