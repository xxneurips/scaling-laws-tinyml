"""
Stage 4A: PyTorch → ONNX → TensorFlow → TFLite conversion pipeline.
=====================================================================
Converts trained PyTorch checkpoints to TFLite format (float32 + int8).
Validates outputs match at each stage (atol 1e-4).

Pipeline: PyTorch → ONNX (opset 12) → TF SavedModel → TFLite (float32)
                                                      → TFLite (int8, full integer)

Usage:
    python convert_tflite.py                                  # All checkpoints
    python convert_tflite.py --dataset cifar100               # Specific dataset
    python convert_tflite.py --dataset cifar100 --arch scalecnn --sample 5
"""

import argparse
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from phase1_train import build_model, get_data_loaders

BASE_DIR = Path(__file__).parent
TFLITE_DIR = BASE_DIR / "tflite_models"


# =============================================================================
# Export to ONNX
# =============================================================================

def export_onnx(model, input_shape, onnx_path, opset=12):
    """Export PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, *input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"    ONNX exported: {onnx_path}")
    return onnx_path


# =============================================================================
# ONNX → TF SavedModel
# =============================================================================

def onnx_to_tf(onnx_path, tf_dir):
    """Convert ONNX model to TensorFlow SavedModel."""
    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(tf_dir))
    print(f"    TF SavedModel exported: {tf_dir}")
    return tf_dir


# =============================================================================
# TF → TFLite
# =============================================================================

def tf_to_tflite_float32(tf_dir, tflite_path):
    """Convert TF SavedModel to TFLite float32."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"    TFLite (float32): {tflite_path} ({size_kb:.1f} KB)")
    return tflite_path, size_kb


def tf_to_tflite_int8(tf_dir, tflite_path, representative_data):
    """Convert TF SavedModel to TFLite int8 (full integer quantization)."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"    TFLite (int8):    {tflite_path} ({size_kb:.1f} KB)")
    return tflite_path, size_kb


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def get_pytorch_output(model, input_tensor):
    """Get PyTorch model output for a given input."""
    model.eval()
    return model(input_tensor).numpy()


def validate_onnx(model, onnx_path, input_shape, atol=1e-4):
    """Validate ONNX output matches PyTorch."""
    import onnxruntime as ort

    dummy = torch.randn(1, *input_shape)
    pt_out = get_pytorch_output(model, dummy)

    session = ort.InferenceSession(str(onnx_path))
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]

    max_diff = np.max(np.abs(pt_out - onnx_out))
    match = max_diff < atol
    print(f"    ONNX validation: max_diff={max_diff:.6f} {'PASS' if match else 'FAIL'}")
    return match, max_diff


def validate_tflite(tflite_path, model, input_shape, atol=1e-4):
    """Validate TFLite float32 output matches PyTorch."""
    import tensorflow as tf

    dummy = torch.randn(1, *input_shape)
    pt_out = get_pytorch_output(model, dummy)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], dummy.numpy())
    interpreter.invoke()
    tflite_out = interpreter.get_tensor(output_details[0]["index"])

    max_diff = np.max(np.abs(pt_out - tflite_out))
    match = max_diff < atol
    print(f"    TFLite validation: max_diff={max_diff:.6f} {'PASS' if match else 'FAIL'}")
    return match, max_diff


# =============================================================================
# Representative dataset generator (for int8 calibration)
# =============================================================================

def make_representative_dataset(loader, input_shape, n_samples=200):
    """Generate representative dataset for int8 quantization calibration."""
    import tensorflow as tf

    samples = []
    for inputs, _ in loader:
        for i in range(inputs.size(0)):
            samples.append(inputs[i].numpy())
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break

    def generator():
        for sample in samples:
            yield [np.expand_dims(sample, axis=0).astype(np.float32)]

    return generator


# =============================================================================
# Main conversion pipeline
# =============================================================================

def convert_checkpoint(ckpt_path, dataset_name, output_dir, test_loader, validate=True):
    """Full conversion pipeline for one checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch")
    size_param = ckpt.get("size_param")
    num_classes = ckpt.get("num_classes", 100)
    input_shape = tuple(ckpt.get("input_shape", (3, 32, 32)))

    if arch is None or size_param is None:
        print(f"  [SKIP] Missing metadata in {ckpt_path.name}")
        return None

    run_name = ckpt_path.stem
    print(f"\n  Converting: {run_name} (arch={arch}, params at size={size_param})")

    # Build and load model
    model, num_params = build_model(arch, size_param, num_classes, input_shape)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    result = {
        "run_name": run_name,
        "arch": arch,
        "size_param": size_param,
        "num_params": num_params,
        "num_classes": num_classes,
        "input_shape": list(input_shape),
    }

    model_dir = output_dir / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: ONNX
        onnx_path = model_dir / f"{run_name}.onnx"
        export_onnx(model, input_shape, onnx_path)

        if validate:
            onnx_ok, onnx_diff = validate_onnx(model, onnx_path, input_shape)
            result["onnx_max_diff"] = float(onnx_diff)
            result["onnx_valid"] = onnx_ok

        # Step 2: TF SavedModel
        tf_dir = model_dir / "tf_saved_model"
        onnx_to_tf(onnx_path, tf_dir)

        # Step 3: TFLite float32
        tflite_f32_path = model_dir / f"{run_name}_float32.tflite"
        _, f32_size = tf_to_tflite_float32(tf_dir, tflite_f32_path)
        result["tflite_float32_size_kb"] = f32_size

        if validate:
            f32_ok, f32_diff = validate_tflite(tflite_f32_path, model, input_shape)
            result["tflite_float32_max_diff"] = float(f32_diff)
            result["tflite_float32_valid"] = f32_ok

        # Step 4: TFLite int8
        tflite_int8_path = model_dir / f"{run_name}_int8.tflite"
        rep_dataset = make_representative_dataset(test_loader, input_shape)
        _, int8_size = tf_to_tflite_int8(tf_dir, tflite_int8_path, rep_dataset)
        result["tflite_int8_size_kb"] = int8_size
        result["compression_ratio"] = f32_size / int8_size if int8_size > 0 else 0

        # Clean up TF SavedModel (large)
        if tf_dir.exists():
            shutil.rmtree(tf_dir)

        result["success"] = True

    except Exception as e:
        print(f"    [ERROR] {e}")
        result["success"] = False
        result["error"] = str(e)

    return result


def select_representative_checkpoints(checkpoint_dir, n_per_arch=7):
    """
    Select ~n_per_arch representative checkpoints per architecture
    (spanning the full parameter range, seed=0 only).
    """
    ckpts = list(checkpoint_dir.glob("*_s0.pt"))
    if not ckpts:
        ckpts = list(checkpoint_dir.glob("*.pt"))

    by_arch = defaultdict(list)
    for p in ckpts:
        name = p.stem
        if "cnn_" in name:
            by_arch["scalecnn"].append(p)
        elif "mob_" in name:
            by_arch["mobilenet"].append(p)
        elif "eff_" in name:
            by_arch["effnet"].append(p)

    selected = []
    for arch, paths in by_arch.items():
        paths = sorted(paths)
        if len(paths) <= n_per_arch:
            selected.extend(paths)
        else:
            # Evenly space across the range
            indices = np.linspace(0, len(paths) - 1, n_per_arch, dtype=int)
            selected.extend([paths[i] for i in indices])

    return selected


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoints to TFLite")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "cifar10", "tinyimagenet", "speechcommands"])
    parser.add_argument("--arch", type=str, default=None,
                        choices=["scalecnn", "mobilenet", "effnet"])
    parser.add_argument("--sample", type=int, default=7,
                        help="Max checkpoints per architecture to convert")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip output validation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Convert a specific checkpoint file")
    args = parser.parse_args()

    dataset_name = args.dataset
    checkpoint_dir = BASE_DIR / "checkpoints" / dataset_name
    output_dir = TFLITE_DIR / dataset_name

    if not checkpoint_dir.exists():
        print(f"[ERROR] No checkpoints at {checkpoint_dir}")
        print("Run phase1_train.py first (checkpoints are now preserved).")
        return

    # Load test data for representative dataset
    _, test_loader, num_classes, input_shape = get_data_loaders(dataset_name)

    # Select checkpoints
    if args.checkpoint:
        ckpt_paths = [Path(args.checkpoint)]
    else:
        ckpt_paths = select_representative_checkpoints(checkpoint_dir, args.sample)

    if args.arch:
        ckpt_paths = [p for p in ckpt_paths if args.arch[:3] in p.stem]

    print(f"Converting {len(ckpt_paths)} checkpoints from {dataset_name}")
    print(f"Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for ckpt_path in sorted(ckpt_paths):
        result = convert_checkpoint(
            ckpt_path, dataset_name, output_dir, test_loader,
            validate=not args.no_validate
        )
        if result:
            all_results.append(result)

    # Save summary
    summary_path = output_dir / "conversion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  Conversion Summary: {dataset_name}")
    print(f"{'='*70}")
    print(f"  {'Run':<30} {'Params':>10} {'F32 (KB)':>10} {'Int8 (KB)':>10} {'Ratio':>8} {'Valid':>6}")
    print(f"  {'-'*80}")

    for r in all_results:
        if r.get("success"):
            name = r["run_name"][:30]
            valid = "Y" if r.get("tflite_float32_valid", True) else "N"
            print(f"  {name:<30} {r['num_params']:>10,} "
                  f"{r.get('tflite_float32_size_kb', 0):>10.1f} "
                  f"{r.get('tflite_int8_size_kb', 0):>10.1f} "
                  f"{r.get('compression_ratio', 0):>8.1f}x {valid:>6}")

    successes = sum(1 for r in all_results if r.get("success"))
    print(f"\n  {successes}/{len(all_results)} successful conversions")
    print(f"  Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
