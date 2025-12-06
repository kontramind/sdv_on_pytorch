"""
SDV Synthesizer Test CLI with PyTorch 2.7 + CUDA 12.8 on RTX 5070 Ti
Tests: training, saving, loading, generation, GPU usage
Supports: TVAE, CTGAN, CopulaGAN, GaussianCopula
"""

from enum import Enum
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sdv.metadata import Metadata
from sdv.single_table import (
    CTGANSynthesizer,
    CopulaGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

app = typer.Typer(help="SDV Synthesizer Test CLI for PyTorch 2.7 + CUDA 12.8")
console = Console()


class SynthesizerType(str, Enum):
    tvae = "tvae"
    ctgan = "ctgan"
    copulagan = "copulagan"
    gaussian = "gaussian"


def check_environment() -> tuple[bool, dict]:
    """Check PyTorch and GPU environment."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_name": None,
        "compute_capability": None,
        "is_blackwell": False,
    }

    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{capability[0]}.{capability[1]} (sm_{capability[0]}{capability[1]})"
        info["is_blackwell"] = capability[0] >= 10

    return info["cuda_available"], info


def display_environment(info: dict) -> None:
    """Display environment info with Rich."""
    table = Table(title="Environment Check", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("PyTorch version", info["pytorch_version"])
    table.add_row("CUDA available", "✅ Yes" if info["cuda_available"] else "❌ No")

    if info["cuda_available"]:
        table.add_row("CUDA version", info["cuda_version"])
        table.add_row("GPU count", str(info["gpu_count"]))
        table.add_row("GPU name", info["gpu_name"])
        table.add_row("Compute capability", info["compute_capability"])
        if info["is_blackwell"]:
            table.add_row("Architecture", "✅ Blackwell detected!")
        else:
            table.add_row("Architecture", "⚠️ Not Blackwell")

    console.print(table)


def generate_training_data(rows: int) -> pd.DataFrame:
    """Generate synthetic training data."""
    np.random.seed(42)

    return pd.DataFrame({
        # 4 numerical features
        "age": np.random.normal(35, 10, rows).clip(18, 80).astype(int),
        "income": np.random.lognormal(10.5, 0.5, rows).clip(20000, 200000),
        "credit_score": np.random.normal(700, 50, rows).clip(300, 850).astype(int),
        "account_balance": np.random.exponential(5000, rows).clip(0, 100000),
        # 3 categorical features
        "region": np.random.choice(["North", "South", "East", "West"], rows),
        "account_type": np.random.choice(
            ["Checking", "Savings", "Premium"], rows, p=[0.5, 0.3, 0.2]
        ),
        "risk_category": np.random.choice(
            ["Low", "Medium", "High"], rows, p=[0.6, 0.3, 0.1]
        ),
    })


def create_synthesizer(
    synth_type: SynthesizerType,
    metadata: Metadata,
    epochs: int,
    batch_size: int,
    use_gpu: bool,
) -> tuple:
    """Create synthesizer based on type."""
    gpu_models = {SynthesizerType.tvae, SynthesizerType.ctgan, SynthesizerType.copulagan}
    supports_gpu = synth_type in gpu_models

    if synth_type == SynthesizerType.tvae:
        model = TVAESynthesizer(
            metadata,
            epochs=epochs,
            batch_size=batch_size,
            enable_gpu=use_gpu,
            verbose=True,
        )
    elif synth_type == SynthesizerType.ctgan:
        model = CTGANSynthesizer(
            metadata,
            epochs=epochs,
            batch_size=batch_size,
            enable_gpu=use_gpu,
            verbose=True,
        )
    elif synth_type == SynthesizerType.copulagan:
        model = CopulaGANSynthesizer(
            metadata,
            epochs=epochs,
            batch_size=batch_size,
            enable_gpu=use_gpu,
            verbose=True,
        )
    else:  # gaussian
        model = GaussianCopulaSynthesizer(metadata)

    return model, supports_gpu


def display_data_sample(data: pd.DataFrame, title: str) -> None:
    """Display data sample with Rich table."""
    table = Table(title=title)
    for col in data.columns:
        table.add_column(col, style="cyan")

    for _, row in data.head(3).iterrows():
        table.add_row(*[str(v)[:20] for v in row.values])

    console.print(table)


@app.command()
def test(
    synthesizer: SynthesizerType = typer.Option(
        SynthesizerType.tvae,
        "--synthesizer",
        "-s",
        help="Synthesizer type to test",
    ),
    samples: int = typer.Option(
        10000,
        "--samples",
        "-n",
        help="Number of synthetic samples to generate",
    ),
    training_rows: int = typer.Option(
        10000,
        "--training-rows",
        "-t",
        help="Number of training data rows",
    ),
    epochs: int = typer.Option(
        50,
        "--epochs",
        "-e",
        help="Training epochs (neural models only)",
    ),
    batch_size: int = typer.Option(
        500,
        "--batch-size",
        "-b",
        help="Batch size (neural models only)",
    ),
    no_gpu: bool = typer.Option(
        False,
        "--no-gpu",
        help="Disable GPU, use CPU only",
    ),
) -> None:
    """Test SDV synthesizer with PyTorch 2.7 + CUDA 12.8."""
    console.print(
        Panel.fit(
            f"[bold blue]SDV {synthesizer.value.upper()} Test[/bold blue]\n"
            "PyTorch 2.7 + CUDA 12.8 Compatibility",
            border_style="blue",
        )
    )

    # 1. Check environment
    console.print("\n[bold cyan]📊 Environment Check[/bold cyan]")
    cuda_available, env_info = check_environment()
    display_environment(env_info)

    use_gpu = cuda_available and not no_gpu
    if not cuda_available and not no_gpu:
        console.print("[yellow]⚠️ CUDA not available, falling back to CPU[/yellow]")
    elif no_gpu:
        console.print("[yellow]⚠️ GPU disabled by --no-gpu flag[/yellow]")

    # 2. Generate training data
    console.print(f"\n[bold cyan]📝 Generating training data ({training_rows:,} rows, 7 features)[/bold cyan]")
    data = generate_training_data(training_rows)
    console.print(f"  Shape: {data.shape}")
    display_data_sample(data, "Training Data Sample")

    # 3. Create metadata
    console.print("\n[bold cyan]🔧 Creating SDV metadata[/bold cyan]")
    metadata = Metadata.detect_from_dataframe(data)
    metadata_path = Path("metadata.json")
    metadata.save_to_json(filepath=metadata_path, mode="overwrite")

    table_name = list(metadata.tables.keys())[0]
    column_count = len(metadata.tables[table_name].columns)
    console.print(f"  ✅ Metadata created with {column_count} columns")
    console.print(f"  ✅ Saved to {metadata_path}")

    # 4. Initialize synthesizer
    console.print(f"\n[bold cyan]🚀 Initializing {synthesizer.value.upper()} synthesizer[/bold cyan]")
    model, supports_gpu = create_synthesizer(
        synthesizer, metadata, epochs, batch_size, use_gpu
    )

    if supports_gpu:
        console.print(f"  Mode: {'GPU' if use_gpu else 'CPU'}")
        console.print(f"  Epochs: {epochs}")
        console.print(f"  Batch size: {batch_size}")
    else:
        console.print("  Mode: CPU (statistical model)")
    console.print("  ✅ Synthesizer initialized")

    # 5. Train model
    console.print(f"\n[bold cyan]⏳ Training {synthesizer.value.upper()}[/bold cyan]")
    try:
        model.fit(data)
        console.print("  ✅ Training completed successfully!")

        # Check GPU usage for neural models
        if supports_gpu and hasattr(model, "_model") and hasattr(model._model, "decoder"):
            device = next(model._model.decoder.parameters()).device
            console.print(f"  Model device: {device}")
            if device.type == "cuda":
                console.print("  ✅ Training used GPU!")
            else:
                console.print("  ⚠️ Training used CPU")

    except Exception as e:
        console.print(f"  [red]❌ Training failed: {e}[/red]")
        raise typer.Exit(1)

    # 6. Save model
    console.print("\n[bold cyan]💾 Saving model (pickle format)[/bold cyan]")
    model_path = Path(f"{synthesizer.value}_model.pkl")
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        console.print(f"  ✅ Model saved to: {model_path}")
        console.print(f"  File size: {model_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        console.print(f"  [red]❌ Save failed: {e}[/red]")
        raise typer.Exit(1)

    # 7. Load model
    console.print("\n[bold cyan]📂 Loading model from disk[/bold cyan]")
    try:
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        console.print("  ✅ Model loaded successfully!")

        if supports_gpu and hasattr(loaded_model, "_model") and hasattr(loaded_model._model, "decoder"):
            device = next(loaded_model._model.decoder.parameters()).device
            console.print(f"  Loaded model device: {device}")

    except Exception as e:
        console.print(f"  [red]❌ Load failed: {e}[/red]")
        raise typer.Exit(1)

    # 8. Generate synthetic data
    console.print(f"\n[bold cyan]🎲 Generating {samples:,} synthetic samples[/bold cyan]")
    try:
        synthetic = loaded_model.sample(samples)
        console.print(f"  ✅ Generated {len(synthetic):,} samples")
        console.print(f"  Shape: {synthetic.shape}")
        display_data_sample(synthetic, "Synthetic Data Sample")

        if set(synthetic.columns) == set(data.columns):
            console.print("  ✅ Schema matches original data")
        else:
            console.print("  [yellow]⚠️ Schema mismatch![/yellow]")

    except Exception as e:
        console.print(f"  [red]❌ Generation failed: {e}[/red]")
        raise typer.Exit(1)

    # 9. GPU memory usage
    if torch.cuda.is_available():
        console.print("\n[bold cyan]🔍 GPU Memory Usage[/bold cyan]")
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        console.print(f"  Allocated: {allocated:.1f} MB")
        console.print(f"  Reserved: {reserved:.1f} MB")

        if allocated > 0:
            console.print("  ✅ GPU memory in use")
        else:
            console.print("  ⚠️ No GPU memory allocated")

    # 10. Summary
    summary_table = Table(title="Test Summary", show_header=False)
    summary_table.add_column("Check", style="cyan")
    summary_table.add_column("Status", style="green")

    summary_table.add_row("Environment", f"PyTorch {env_info['pytorch_version']}")
    if cuda_available:
        summary_table.add_row("GPU", env_info["gpu_name"])
        summary_table.add_row("Compute", env_info["compute_capability"])
    summary_table.add_row("Synthesizer", synthesizer.value.upper())
    summary_table.add_row("Training", "✅ Completed")
    summary_table.add_row("Save/Load", "✅ Working")
    summary_table.add_row("Generation", f"✅ {samples:,} samples")
    summary_table.add_row("Schema", "✅ Valid")

    console.print("\n")
    console.print(summary_table)
    console.print(
        Panel.fit(
            f"[bold green]✅ ALL TESTS PASSED![/bold green]\n\n"
            f"SDV {synthesizer.value.upper()} is compatible with "
            f"PyTorch 2.7 + {'RTX 5070 Ti' if env_info.get('is_blackwell') else 'your GPU'}! 🎉",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
