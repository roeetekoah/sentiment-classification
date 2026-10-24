"""
Forward-only computational-cost benchmark for the four Exercise 5 models.

The benchmark uses batch size 64, five warm-up iterations, and 50 timed model
forward passes. It runs every model in eval mode under torch.no_grad(), with
CUDA synchronization immediately before and after timing. Reported time is
amortized per sentence; tokenization, one-hot construction, frozen-embedding
lookup/averaging, data transfer, model loading, and all training are excluded.

Run from code/:
    python benchmark.py --device cuda
"""

import argparse
import platform
import time

import torch

import main as exercise

try:
    from transformers.utils import logging as transformer_logging

    transformer_logging.set_verbosity_error()
    transformer_logging.disable_progress_bar()
except (ImportError, AttributeError):
    pass


BATCH_SIZE = 64
WARMUP_ITERATIONS = 5
TIMED_ITERATIONS = 50
FROZEN_EMBEDDING_VALUES = 50_265 * 768
FP32_BYTES = 4

EXPECTED_PARAMETERS = {
    "one-hot": 16_272,
    "embedding": 769,
    "BiLSTM": 696_201,
    "fine-tuned": 82_119_169,
}


def trainable_parameters(model):
    """Count only values whose requires_grad flag is true."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def benchmark_forward(model, inputs, device):
    """Return seconds per batch for repeated model-forward calls."""
    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            model(*inputs)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()

        for _ in range(TIMED_ITERATIONS):
            model(*inputs)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

    return (time.perf_counter() - started) / TIMED_ITERATIONS


def hardware_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine()


def build_cases(device):
    """Construct models and already-preprocessed synthetic model inputs."""
    return (
        (
            "one-hot",
            exercise.LogLinear(16_271),
            (torch.randn(BATCH_SIZE, 16_271, device=device),),
        ),
        (
            "embedding",
            exercise.LogLinear(768),
            (torch.randn(BATCH_SIZE, 768, device=device),),
        ),
        (
            "BiLSTM",
            exercise.LSTM(768, 100, 1, 0.5),
            (torch.randn(BATCH_SIZE, exercise.SEQ_LEN, 768, device=device),),
        ),
        (
            "fine-tuned",
            exercise.TransformerSentimentClassifier(),
            (
                torch.randint(0, 50_000, (BATCH_SIZE, 64), device=device),
                torch.ones(BATCH_SIZE, 64, dtype=torch.long, device=device),
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()

    device = (torch.device(args.device) if args.device
              else exercise.get_available_device())
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    exercise.set_seed(42)

    print(f"hardware={hardware_name(device)}")
    print(
        f"python={platform.python_version()}  torch={torch.__version__}  "
        f"cuda={torch.version.cuda}  cudnn={torch.backends.cudnn.version()}"
    )
    print(
        f"device={device.type}  batch={BATCH_SIZE}  "
        f"warmup={WARMUP_ITERATIONS}  timed_iterations={TIMED_ITERATIONS}"
    )
    print(
        f"frozen_embedding={FROZEN_EMBEDDING_VALUES:,} FP32 values  "
        f"storage_MB={FROZEN_EMBEDDING_VALUES * FP32_BYTES / 1_000_000:.3f}"
    )
    print()
    print(f"{'model':12s} {'trainable':>12s} {'ms/batch':>10s} "
          f"{'us/sentence':>12s}")

    for name, model, inputs in build_cases(device):
        model.to(device)
        count = trainable_parameters(model)
        if count != EXPECTED_PARAMETERS[name]:
            raise AssertionError(
                f"{name}: expected {EXPECTED_PARAMETERS[name]:,}, found {count:,}"
            )
        milliseconds = benchmark_forward(model, inputs, device) * 1_000
        print(
            f"{name:12s} {count:12,d} {milliseconds:10.3f} "
            f"{milliseconds * 1_000 / BATCH_SIZE:12.1f}"
        )

    full_values = EXPECTED_PARAMETERS["fine-tuned"]
    print()
    print(
        f"fine_tuned_FP32_parameter_storage_MB="
        f"{full_values * FP32_BYTES / 1_000_000:.3f}"
    )
    print("us/sentence is amortized batch-64 throughput, not batch-1 latency.")
    print("Timing covers model forward only; preprocessing and data transfer excluded.")


if __name__ == "__main__":
    main()
