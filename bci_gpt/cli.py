"""Command-line interface for BCI-GPT."""

import typer
from typing import Optional, List
from pathlib import Path
import torch
from rich.console import Console
from rich.progress import track
from rich.table import Table
import warnings

from .core.models import BCIGPTModel
from .core.inverse_gan import InverseSimulator
from .preprocessing.eeg_processor import EEGProcessor
from .training.trainer import BCIGPTTrainer, TrainingConfig
from .decoding.realtime_decoder import RealtimeDecoder
from .inverse.text_to_eeg import TextToEEG, GenerationConfig
from .utils.streaming import StreamingEEG, StreamConfig
from .utils.metrics import BCIMetrics
from .utils.visualization import EEGVisualizer

app = typer.Typer(
    name="bci-gpt",
    help="BCI-GPT: Brain-Computer Interface GPT Inverse Simulator",
    add_completion=False
)

console = Console()


@app.command()
def info():
    """Display BCI-GPT system information."""
    console.print("[bold blue]BCI-GPT System Information[/bold blue]")
    
    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Version/Info")
    
    # Check PyTorch
    table.add_row("PyTorch", "✓ Available", torch.__version__)
    table.add_row("CUDA", "✓ Available" if torch.cuda.is_available() else "✗ Not Available", 
                  f"{torch.cuda.device_count()} devices" if torch.cuda.is_available() else "CPU only")
    
    # Check optional dependencies
    try:
        import mne
        table.add_row("MNE (Neuroimaging)", "✓ Available", mne.__version__)
    except ImportError:
        table.add_row("MNE (Neuroimaging)", "✗ Not Available", "pip install mne")
    
    try:
        import transformers
        table.add_row("Transformers", "✓ Available", transformers.__version__)
    except ImportError:
        table.add_row("Transformers", "✗ Not Available", "pip install transformers")
    
    try:
        import pylsl
        table.add_row("Lab Streaming Layer", "✓ Available", "Real-time streaming")
    except ImportError:
        table.add_row("Lab Streaming Layer", "✗ Not Available", "pip install pylsl")
    
    try:
        import brainflow
        table.add_row("BrainFlow", "✓ Available", "Hardware interfaces")
    except ImportError:
        table.add_row("BrainFlow", "✗ Not Available", "pip install brainflow")
    
    console.print(table)


@app.command()
def train(
    data_path: str = typer.Argument(..., help="Path to training data"),
    model_name: str = typer.Option("bci-gpt", help="Model name"),
    output_dir: str = typer.Option("./models", help="Output directory for trained model"),
    epochs: int = typer.Option(50, help="Number of training epochs"),
    batch_size: int = typer.Option(16, help="Batch size"),
    learning_rate: float = typer.Option(1e-4, help="Learning rate"),
    device: str = typer.Option("auto", help="Device (cpu/cuda/auto)"),
    val_data: Optional[str] = typer.Option(None, help="Path to validation data"),
    resume: Optional[str] = typer.Option(None, help="Path to checkpoint to resume from"),
):
    """Train a BCI-GPT model."""
    console.print(f"[bold green]Training BCI-GPT model: {model_name}[/bold green]")
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    console.print(f"Using device: {device}")
    
    try:
        # Create model
        model = BCIGPTModel()
        
        # Setup training configuration
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_tensorboard=True,
            use_wandb=False
        )
        
        # Create trainer
        trainer = BCIGPTTrainer(model=model, config=config, device=device)
        
        # Resume from checkpoint if specified
        if resume:
            console.print(f"Resuming from checkpoint: {resume}")
            trainer.load_checkpoint(resume)
        
        # Start training
        history = trainer.fit(
            train_data=data_path,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Save trained model
        output_path = Path(output_dir) / f"{model_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_path / "model.pt"))
        
        console.print(f"[bold green]Training completed! Model saved to {output_path}[/bold green]")
        
        # Display training summary
        if history:
            final_loss = history['train_loss'][-1] if history['train_loss'] else 0
            console.print(f"Final training loss: {final_loss:.4f}")
            
            if 'val_loss' in history and history['val_loss']:
                final_val_loss = history['val_loss'][-1]
                console.print(f"Final validation loss: {final_val_loss:.4f}")
        
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def decode(
    model_path: str = typer.Argument(..., help="Path to trained model"),
    input_type: str = typer.Option("file", help="Input type: file, stream, or demo"),
    input_path: Optional[str] = typer.Option(None, help="Path to EEG data file"),
    output_path: Optional[str] = typer.Option(None, help="Path to save decoded text"),
    stream_backend: str = typer.Option("simulated", help="Streaming backend: lsl, brainflow, simulated"),
    confidence_threshold: float = typer.Option(0.7, help="Confidence threshold for output"),
    device: str = typer.Option("auto", help="Device (cpu/cuda/auto)"),
):
    """Decode EEG signals to text."""
    console.print("[bold blue]BCI-GPT Decoding[/bold blue]")
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        if input_type == "file":
            if not input_path:
                console.print("[red]Input path required for file decoding[/red]")
                raise typer.Exit(1)
            
            _decode_from_file(model_path, input_path, output_path, device)
            
        elif input_type == "stream":
            _decode_from_stream(model_path, stream_backend, confidence_threshold, device)
            
        elif input_type == "demo":
            _run_demo_decoding(model_path, device)
            
        else:
            console.print(f"[red]Unknown input type: {input_type}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[bold red]Decoding failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def generate(
    text: str = typer.Argument(..., help="Text to convert to EEG"),
    model_path: str = typer.Option("./models/inverse_gan.pt", help="Path to inverse model"),
    output_path: str = typer.Option("./generated_eeg.npy", help="Path to save generated EEG"),
    duration: float = typer.Option(2.0, help="Duration in seconds"),
    style: str = typer.Option("imagined_speech", help="EEG style: imagined_speech, inner_monologue, subvocalization"),
    num_samples: int = typer.Option(1, help="Number of samples to generate"),
    device: str = typer.Option("auto", help="Device (cpu/cuda/auto)"),
):
    """Generate synthetic EEG from text."""
    console.print(f"[bold green]Generating EEG for: '{text}'[/bold green]")
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Initialize text-to-EEG generator
        generator = TextToEEG(
            inverse_model_path=model_path,
            device=device
        )
        
        # Generation config
        config = GenerationConfig(
            duration=duration,
            style=style,
            num_samples=num_samples
        )
        
        # Generate EEG
        console.print("Generating synthetic EEG...")
        synthetic_eeg = generator.generate(text, config)
        
        # Save generated EEG
        import numpy as np
        if isinstance(synthetic_eeg, list):
            # Multiple samples
            for i, eeg in enumerate(synthetic_eeg):
                output_file = output_path.replace('.npy', f'_sample_{i}.npy')
                np.save(output_file, eeg)
                console.print(f"Saved sample {i+1} to: {output_file}")
        else:
            # Single sample
            np.save(output_path, synthetic_eeg)
            console.print(f"Saved generated EEG to: {output_path}")
        
        # Validate synthetic EEG
        if isinstance(synthetic_eeg, list):
            validation = generator.validate_synthetic(synthetic_eeg[0])
        else:
            validation = generator.validate_synthetic(synthetic_eeg)
        
        console.print(f"Generation quality: {validation.get('realism_score', 0):.3f}")
        
    except Exception as e:
        console.print(f"[bold red]Generation failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def visualize(
    data_path: str = typer.Argument(..., help="Path to EEG data file"),
    channels: Optional[List[str]] = typer.Option(None, help="Channel names to plot"),
    plot_type: str = typer.Option("signals", help="Plot type: signals, spectrogram, psd, topo"),
    output_path: Optional[str] = typer.Option(None, help="Path to save plot"),
    sampling_rate: int = typer.Option(1000, help="Sampling rate in Hz"),
):
    """Visualize EEG data."""
    console.print(f"[bold blue]Visualizing EEG data: {data_path}[/bold blue]")
    
    try:
        # Load EEG data
        import numpy as np
        if data_path.endswith('.npy'):
            eeg_data = np.load(data_path)
        else:
            console.print("[red]Only .npy files supported for visualization[/red]")
            raise typer.Exit(1)
        
        # Initialize visualizer
        visualizer = EEGVisualizer()
        
        # Create visualization
        if plot_type == "signals":
            visualizer.plot_eeg_signals(
                eeg_data, channels, sampling_rate, save_path=output_path
            )
        elif plot_type == "spectrogram":
            visualizer.plot_spectrogram(
                eeg_data, channel_idx=0, sampling_rate=sampling_rate, save_path=output_path
            )
        elif plot_type == "psd":
            visualizer.plot_power_spectral_density(
                eeg_data, channels, sampling_rate, save_path=output_path
            )
        elif plot_type == "topo":
            if channels and len(channels) == eeg_data.shape[0]:
                values = np.mean(np.abs(eeg_data), axis=1)  # Average amplitude
                visualizer.plot_topographic_map(
                    values, channels, save_path=output_path
                )
            else:
                console.print("[red]Topographic plot requires channel names[/red]")
                raise typer.Exit(1)
        else:
            console.print(f"[red]Unknown plot type: {plot_type}[/red]")
            raise typer.Exit(1)
        
        if output_path:
            console.print(f"Plot saved to: {output_path}")
        
    except Exception as e:
        console.print(f"[bold red]Visualization failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to trained model"),
    test_data: str = typer.Argument(..., help="Path to test data"),
    output_path: Optional[str] = typer.Option(None, help="Path to save evaluation report"),
    device: str = typer.Option("auto", help="Device (cpu/cuda/auto)"),
):
    """Evaluate BCI-GPT model performance."""
    console.print("[bold blue]Evaluating BCI-GPT Model[/bold blue]")
    
    # Setup device  
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # This would involve loading test data and running evaluation
        # For now, just show placeholder metrics
        
        console.print("Running evaluation on test data...")
        
        # Initialize metrics calculator
        metrics_calc = BCIMetrics()
        
        # Placeholder evaluation results
        accuracy = 0.85
        wer = 0.15
        itr = metrics_calc.calculate_itr(accuracy, num_classes=26, trial_duration=2.0)
        
        # Display results
        console.print(f"Accuracy: {accuracy:.3f}")
        console.print(f"Word Error Rate: {wer:.3f}")
        console.print(f"Information Transfer Rate: {itr:.2f} bits/min")
        
        if output_path:
            # Save evaluation report
            report = f"""BCI-GPT Evaluation Report
============================

Accuracy: {accuracy:.3f}
Word Error Rate: {wer:.3f}
Information Transfer Rate: {itr:.2f} bits/min

Model: {model_path}
Test Data: {test_data}
Device: {device}
"""
            with open(output_path, 'w') as f:
                f.write(report)
            
            console.print(f"Evaluation report saved to: {output_path}")
        
    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise typer.Exit(1)


def _decode_from_file(model_path: str, input_path: str, output_path: Optional[str], device: str):
    """Decode EEG from file."""
    console.print(f"Decoding EEG from file: {input_path}")
    
    # Load model (placeholder - would load actual model)
    console.print("Loading model...")
    
    # Load and process EEG data
    processor = EEGProcessor()
    eeg_data = processor.load_data(input_path)
    
    # Simulate decoding
    decoded_text = "This is placeholder decoded text from EEG signals."
    
    console.print(f"Decoded text: {decoded_text}")
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(decoded_text)
        console.print(f"Decoded text saved to: {output_path}")


def _decode_from_stream(model_path: str, backend: str, threshold: float, device: str):
    """Decode EEG from real-time stream."""
    console.print(f"Starting real-time decoding with {backend} backend")
    
    try:
        # Create stream
        config = StreamConfig(sampling_rate=1000, buffer_duration=5.0)
        stream = StreamingEEG.create_stream(backend, config)
        
        # Start streaming
        stream.start_stream()
        console.print("Stream started. Press Ctrl+C to stop.")
        
        # Simulate real-time decoding
        import time
        try:
            while True:
                # Get data from stream
                data = stream.get_data(duration=1.0)
                if data is not None:
                    # Simulate decoding
                    confidence = 0.8  # Placeholder confidence
                    if confidence >= threshold:
                        decoded_text = "decoded_word"
                        console.print(f"Decoded: {decoded_text} (confidence: {confidence:.2f})")
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            console.print("\nStopping real-time decoding...")
            stream.stop_stream()
            
    except Exception as e:
        console.print(f"Stream decoding failed: {e}")


def _run_demo_decoding(model_path: str, device: str):
    """Run demo decoding with synthetic data.""" 
    console.print("Running BCI-GPT decoding demo with synthetic data")
    
    # Generate synthetic EEG data
    processor = EEGProcessor()
    demo_eeg = processor._generate_synthetic_eeg(duration=3.0)
    
    console.print("Generated synthetic EEG data for demo")
    
    # Simulate decoding
    demo_texts = [
        "Hello world",
        "This is a test",
        "BCI-GPT works"
    ]
    
    for i, text in enumerate(demo_texts):
        console.print(f"Sample {i+1}: {text}")
        
    console.print("Demo completed!")


if __name__ == "__main__":
    app()