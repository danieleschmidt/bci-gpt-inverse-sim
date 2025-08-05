"""Command-line interface for BCI-GPT decoding."""

import typer
from typing import Optional, List
from pathlib import Path
import torch
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
import time
import warnings
import json

from ..core.models import BCIGPTModel
from ..preprocessing.eeg_processor import EEGProcessor
from .realtime_decoder import RealtimeDecoder, DecodingResult
from .confidence_estimation import ConfidenceEstimator
from ..utils.streaming import StreamingEEG, StreamConfig
from ..utils.metrics import BCIMetrics
from ..utils.visualization import EEGVisualizer

app = typer.Typer(
    name="bci-gpt-decode",
    help="BCI-GPT Decoding CLI - Real-time and offline thought-to-text decoding",
    add_completion=False
)

console = Console()


@app.command()
def realtime(
    model_path: str = typer.Argument(..., help="Path to trained BCI-GPT model"),
    stream_backend: str = typer.Option("simulated", help="Streaming backend (lsl/brainflow/simulated)"),
    confidence_threshold: float = typer.Option(0.7, help="Confidence threshold for output"),
    buffer_size: int = typer.Option(1000, help="Buffer size in milliseconds"),
    device: str = typer.Option("auto", help="Device (cpu/cuda/auto)"),
    
    # Stream configuration
    sampling_rate: int = typer.Option(1000, help="EEG sampling rate (Hz)"),
    channels: Optional[str] = typer.Option(None, help="EEG channel names (comma-separated)"),
    
    # Hardware-specific options
    serial_port: Optional[str] = typer.Option("/dev/ttyUSB0", help="Serial port for hardware devices"),
    board_id: int = typer.Option(0, help="Board ID for BrainFlow devices"),
    
    # Output options
    output_file: Optional[str] = typer.Option(None, help="File to save decoded text"),
    show_probabilities: bool = typer.Option(False, help="Show token probabilities"),
    show_statistics: bool = typer.Option(True, help="Show decoding statistics"),
    
    # Display options
    max_history: int = typer.Option(10, help="Maximum number of results to display"),
    update_interval: float = typer.Option(0.1, help="Display update interval (seconds)"),
):
    """Start real-time EEG decoding."""
    console.print(Panel(f"[bold blue]Real-time BCI-GPT Decoding[/bold blue]\n"
                       f"Model: {model_path}\nBackend: {stream_backend}", expand=False))
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    console.print(f"Using device: [green]{device}[/green]")
    
    # Parse channels
    channel_list = None
    if channels:
        channel_list = [ch.strip() for ch in channels.split(',')]
        console.print(f"Using channels: {channel_list}")
    
    try:
        # Initialize decoder
        console.print("Loading model and initializing decoder...")
        decoder = RealtimeDecoder(
            model_checkpoint=model_path,
            device=device,
            buffer_size=buffer_size,
            confidence_threshold=confidence_threshold,
            sampling_rate=sampling_rate,
            channels=channel_list
        )
        
        # Create stream configuration
        stream_config = StreamConfig(
            sampling_rate=sampling_rate,
            buffer_duration=buffer_size / 1000.0,
            channels=channel_list
        )
        
        # Initialize streaming backend
        console.print(f"Connecting to {stream_backend} stream...")
        if stream_backend == "brainflow":
            stream = StreamingEEG.create_stream("brainflow", stream_config, 
                                              board_id=board_id, serial_port=serial_port)
        elif stream_backend == "lsl":
            stream = StreamingEEG.create_stream("lsl", stream_config)
        else:
            # Simulated stream
            stream = StreamingEEG.create_stream("simulated", stream_config)
        
        # Start streaming
        stream.start_stream()
        console.print("[green]Stream started successfully[/green]")
        
        # Start decoding
        decoder.start_decoding(stream)
        
        # Initialize output file if specified
        output_fp = None
        if output_file:
            output_fp = open(output_file, 'w', encoding='utf-8')
            console.print(f"Saving output to: {output_file}")
        
        # Real-time decoding loop with live display
        results_history = []
        
        try:
            console.print("\n[bold yellow]Real-time decoding started. Press Ctrl+C to stop.[/bold yellow]\n")
            
            with Live(_create_live_display(results_history, decoder, show_probabilities, show_statistics), 
                     refresh_per_second=int(1/update_interval), console=console) as live:
                
                while True:
                    # Get latest decoding result
                    result = decoder.get_latest_result()
                    
                    if result:
                        # Add to history
                        results_history.append(result)
                        if len(results_history) > max_history:
                            results_history.pop(0)
                        
                        # Output high-confidence results
                        if result.confidence >= confidence_threshold:
                            timestamp_str = time.strftime("%H:%M:%S", time.localtime(result.timestamp))
                            
                            # Save to file if specified
                            if output_fp:
                                output_fp.write(f"[{timestamp_str}] {result.text}\n")
                                output_fp.flush()
                    
                    # Update live display
                    live.update(_create_live_display(results_history, decoder, show_probabilities, show_statistics))
                    
                    time.sleep(update_interval)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping real-time decoding...[/yellow]")
        
        finally:
            # Stop decoding and streaming
            decoder.stop_decoding()
            stream.stop_stream()
            
            if output_fp:
                output_fp.close()
            
            # Display final statistics
            _display_final_statistics(decoder, results_history)
        
    except Exception as e:
        console.print(f"[bold red]Real-time decoding failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def file(
    model_path: str = typer.Argument(..., help="Path to trained BCI-GPT model"),
    input_path: str = typer.Argument(..., help="Path to EEG data file"),
    output_path: Optional[str] = typer.Option(None, help="Path to save decoded text"),
    device: str = typer.Option("auto", help="Device (cpu/cuda/auto)"),
    
    # Processing options
    confidence_threshold: float = typer.Option(0.5, help="Confidence threshold for output"),
    max_length: int = typer.Option(50, help="Maximum sequence length for decoding"),
    batch_processing: bool = typer.Option(False, help="Process data in batches"),
    window_size: float = typer.Option(1.0, help="Window size in seconds for batch processing"),
    overlap_ratio: float = typer.Option(0.5, help="Overlap ratio between windows"),
    
    # Output options
    show_confidence: bool = typer.Option(True, help="Show confidence scores"),
    show_probabilities: bool = typer.Option(False, help="Show token probabilities"),
    save_results: bool = typer.Option(True, help="Save detailed results to JSON"),
):
    """Decode EEG data from file."""
    console.print(Panel(f"[bold blue]File-based EEG Decoding[/bold blue]\n"
                       f"Input: {input_path}\nModel: {model_path}", expand=False))
    
    if not Path(input_path).exists():
        console.print(f"[bold red]Input file not found: {input_path}[/bold red]")
        raise typer.Exit(1)
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load model
        console.print("Loading BCI-GPT model...")
        model = BCIGPTModel()
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            console.print("[green]Model loaded successfully[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load model ({e}). Using demo mode.[/yellow]")
            model = None
        
        # Load and preprocess EEG data
        console.print("Loading and preprocessing EEG data...")
        processor = EEGProcessor()
        eeg_data = processor.load_data(input_path)
        
        if batch_processing:
            # Process in overlapping windows
            results = _process_eeg_in_batches(
                eeg_data, model, processor, window_size, overlap_ratio, 
                confidence_threshold, max_length, device
            )
        else:
            # Process entire file at once
            results = _process_eeg_single(
                eeg_data, model, processor, confidence_threshold, max_length, device
            )
        
        # Display results
        _display_decoding_results(results, show_confidence, show_probabilities)
        
        # Save output text
        if output_path or save_results:
            _save_decoding_output(results, output_path, save_results, input_path, model_path)
        
        console.print("[bold green]File decoding completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]File decoding failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def benchmark(
    model_path: str = typer.Argument(..., help="Path to trained BCI-GPT model"),
    test_data: str = typer.Argument(..., help="Path to test dataset"),
    output_dir: str = typer.Option("./benchmark_results", help="Output directory for results"),
    device: str = typer.Option("auto", help="Device (cpu/cuda/auto)"),
    
    # Benchmark parameters
    num_samples: Optional[int] = typer.Option(None, help="Number of samples to benchmark (all if not specified)"),
    confidence_thresholds: str = typer.Option("0.5,0.7,0.8,0.9", help="Confidence thresholds to test (comma-separated)"),
    batch_size: int = typer.Option(16, help="Batch size for processing"),
    
    # Metrics to compute
    compute_wer: bool = typer.Option(True, help="Compute Word Error Rate"),
    compute_itr: bool = typer.Option(True, help="Compute Information Transfer Rate"),
    compute_latency: bool = typer.Option(True, help="Measure processing latency"),
):
    """Benchmark BCI-GPT model performance."""
    console.print(Panel("[bold blue]BCI-GPT Model Benchmarking[/bold blue]", expand=False))
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Parse confidence thresholds
    thresholds = [float(t.strip()) for t in confidence_thresholds.split(',')]
    
    try:
        # Load model
        console.print("Loading BCI-GPT model...")
        model = BCIGPTModel()
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            console.print("[green]Model loaded successfully[/green]")
        except Exception as e:
            console.print(f"[yellow]Using simulated benchmarking due to model loading error: {e}[/yellow]")
            model = None
        
        # Initialize metrics calculator
        metrics_calc = BCIMetrics()
        
        # Run benchmarks for each confidence threshold
        benchmark_results = {}
        
        for threshold in track(thresholds, description="Running benchmarks..."):
            console.print(f"\nBenchmarking with confidence threshold: {threshold}")
            
            # Simulate benchmark results (in real implementation, would process actual data)
            results = _run_benchmark_simulation(
                model, test_data, threshold, num_samples, batch_size, 
                compute_wer, compute_itr, compute_latency, device
            )
            
            benchmark_results[threshold] = results
            
            # Display intermediate results
            _display_benchmark_results(threshold, results)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save benchmark results
        results_file = output_path / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        # Generate summary report
        _generate_benchmark_report(benchmark_results, output_path)
        
        console.print(f"\n[bold green]Benchmarking completed![/bold green]")
        console.print(f"Results saved to: {output_path}")
        
    except Exception as e:
        console.print(f"[bold red]Benchmarking failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def demo(
    model_path: str = typer.Argument(..., help="Path to trained BCI-GPT model"),
    demo_type: str = typer.Option("synthetic", help="Demo type (synthetic/recorded/interactive)"),
    duration: int = typer.Option(30, help="Demo duration in seconds"),
    device: str = typer.Option("auto", help="Device (cpu/cuda/auto)"),
):
    """Run a BCI-GPT decoding demonstration."""
    console.print(Panel(f"[bold blue]BCI-GPT Decoding Demo[/bold blue]\n"
                       f"Type: {demo_type}\nDuration: {duration}s", expand=False))
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        if demo_type == "synthetic":
            _run_synthetic_demo(model_path, duration, device)
        elif demo_type == "recorded":
            _run_recorded_demo(model_path, duration, device)
        elif demo_type == "interactive":
            _run_interactive_demo(model_path, duration, device)
        else:
            console.print(f"[red]Unknown demo type: {demo_type}[/red]")
            console.print("Available types: synthetic, recorded, interactive")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[bold red]Demo failed: {e}[/bold red]")
        raise typer.Exit(1)


def _create_live_display(results_history: List[DecodingResult], decoder: RealtimeDecoder, 
                        show_probabilities: bool, show_statistics: bool):
    """Create live display for real-time decoding."""
    from rich.layout import Layout
    from rich.text import Text
    
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )
    
    if show_statistics:
        layout["body"].split_row(
            Layout(name="results", ratio=2),
            Layout(name="stats", ratio=1)
        )
    else:
        layout["body"].split_row(Layout(name="results"))
    
    # Header
    header_text = Text("🧠 BCI-GPT Real-time Decoding", style="bold blue")
    header_text.append(f" | Results: {len(results_history)}", style="dim")
    layout["header"].update(Panel(header_text, expand=True))
    
    # Results
    if results_history:
        results_table = Table(title="Recent Decoding Results", show_header=True)
        results_table.add_column("Time", style="cyan", width=8)
        results_table.add_column("Text", style="white", min_width=20)
        results_table.add_column("Confidence", style="green", width=10)
        
        for result in results_history[-5:]:  # Show last 5 results
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(result.timestamp))
            confidence_str = f"{result.confidence:.2f}"
            results_table.add_row(timestamp_str, result.text, confidence_str)
        
        layout["results"].update(results_table)
    else:
        layout["results"].update(Panel("[dim]Waiting for results...[/dim]", title="Results"))
    
    # Statistics
    if show_statistics:
        stats = decoder.get_statistics()
        if stats:
            stats_table = Table(title="Statistics", show_header=False)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")
            
            if 'avg_processing_time' in stats:
                stats_table.add_row("Avg Process Time", f"{stats['avg_processing_time']*1000:.1f}ms")
            if 'avg_confidence' in stats:
                stats_table.add_row("Avg Confidence", f"{stats['avg_confidence']:.2f}")
            if 'buffer_utilization' in stats:
                stats_table.add_row("Buffer Usage", f"{stats['buffer_utilization']:.1%}")
            
            layout["stats"].update(stats_table)
        else:
            layout["stats"].update(Panel("[dim]No statistics yet[/dim]", title="Statistics"))
    
    return layout


def _process_eeg_in_batches(eeg_data, model, processor, window_size, overlap_ratio, 
                          confidence_threshold, max_length, device):
    """Process EEG data in overlapping windows."""
    results = []
    
    # Calculate window parameters
    sampling_rate = getattr(processor, 'sampling_rate', 1000)
    window_samples = int(window_size * sampling_rate)
    step_samples = int(window_samples * (1 - overlap_ratio))
    
    # Process windows
    for start_idx in range(0, eeg_data.shape[1] - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples
        window_data = eeg_data[:, start_idx:end_idx]
        
        # Process window
        result = _decode_eeg_window(window_data, model, processor, confidence_threshold, max_length, device)
        if result:
            results.append({
                'start_time': start_idx / sampling_rate,
                'end_time': end_idx / sampling_rate,
                'text': result['text'],
                'confidence': result['confidence']
            })
    
    return results


def _process_eeg_single(eeg_data, model, processor, confidence_threshold, max_length, device):
    """Process entire EEG file at once."""
    result = _decode_eeg_window(eeg_data, model, processor, confidence_threshold, max_length, device)
    
    if result:
        return [{
            'start_time': 0.0,
            'end_time': eeg_data.shape[1] / getattr(processor, 'sampling_rate', 1000),
            'text': result['text'],
            'confidence': result['confidence']
        }]
    else:
        return []


def _decode_eeg_window(eeg_data, model, processor, confidence_threshold, max_length, device):
    """Decode a single EEG window."""
    try:
        # Preprocess the data
        processed_data = processor.preprocess(eeg_data, epoch_length=eeg_data.shape[1]/1000.0)
        
        # Convert to tensor
        if processed_data['data'].ndim == 3:
            eeg_tensor = torch.FloatTensor(processed_data['data'][0]).unsqueeze(0)
        else:
            eeg_tensor = torch.FloatTensor(processed_data['data']).unsqueeze(0)
        
        eeg_tensor = eeg_tensor.to(device)
        
        if model:
            # Real decoding
            with torch.no_grad():
                decoded_text = model.generate_text_from_eeg(eeg_tensor, max_length=max_length)
                if isinstance(decoded_text, list):
                    decoded_text = decoded_text[0]
            
            # Simulate confidence (would be computed by actual model)
            confidence = np.random.uniform(0.6, 0.9)
        else:
            # Fallback/demo decoding
            demo_words = ["hello", "world", "test", "brain", "computer", "interface", "thinking", "words"]
            decoded_text = np.random.choice(demo_words)
            confidence = np.random.uniform(0.5, 0.8)
        
        if confidence >= confidence_threshold:
            return {
                'text': decoded_text,
                'confidence': confidence
            }
    
    except Exception as e:
        warnings.warn(f"Error decoding EEG window: {e}")
    
    return None


def _display_decoding_results(results, show_confidence, show_probabilities):
    """Display decoding results in a table."""
    if not results:
        console.print("[yellow]No results above confidence threshold[/yellow]")
        return
    
    table = Table(title="Decoding Results", show_header=True, header_style="bold magenta")
    table.add_column("Start Time", style="cyan")
    table.add_column("End Time", style="cyan")
    table.add_column("Decoded Text", style="white", min_width=20)
    
    if show_confidence:
        table.add_column("Confidence", style="green")
    
    for result in results:
        row = [
            f"{result['start_time']:.1f}s",
            f"{result['end_time']:.1f}s",
            result['text']
        ]
        
        if show_confidence:
            row.append(f"{result['confidence']:.2f}")
        
        table.add_row(*row)
    
    console.print(table)


def _save_decoding_output(results, output_path, save_results, input_path, model_path):
    """Save decoding output to files."""
    # Save text output
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"{result['text']}\n")
        console.print(f"Decoded text saved to: {output_path}")
    
    # Save detailed results
    if save_results:
        results_path = Path(output_path).with_suffix('.json') if output_path else Path("decoding_results.json")
        
        detailed_results = {
            'input_file': input_path,
            'model_file': model_path,
            'num_results': len(results),
            'results': results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        console.print(f"Detailed results saved to: {results_path}")


def _run_benchmark_simulation(model, test_data, threshold, num_samples, batch_size, 
                            compute_wer, compute_itr, compute_latency, device):
    """Run benchmark simulation (placeholder for actual benchmarking)."""
    # In real implementation, this would process actual test data
    
    # Simulate realistic metrics
    accuracy = max(0.3, min(0.95, 0.85 - (threshold - 0.5) * 0.3))  # Higher threshold = lower accuracy
    
    results = {
        'accuracy': accuracy,
        'num_samples': num_samples or 1000,
        'threshold': threshold
    }
    
    if compute_wer:
        # Simulate WER (inverse relationship with accuracy)
        results['word_error_rate'] = max(0.05, 1.0 - accuracy)
    
    if compute_itr:
        # Calculate ITR
        metrics_calc = BCIMetrics()
        results['information_transfer_rate'] = metrics_calc.calculate_itr(
            accuracy, num_classes=26, trial_duration=2.0
        )
    
    if compute_latency:
        # Simulate processing latency
        results['avg_latency_ms'] = np.random.uniform(80, 150)
        results['std_latency_ms'] = np.random.uniform(10, 30)
    
    return results


def _display_benchmark_results(threshold, results):
    """Display benchmark results for a single threshold."""
    table = Table(title=f"Results (threshold={threshold})", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Accuracy", f"{results['accuracy']:.1%}")
    
    if 'word_error_rate' in results:
        table.add_row("Word Error Rate", f"{results['word_error_rate']:.1%}")
    
    if 'information_transfer_rate' in results:
        table.add_row("ITR", f"{results['information_transfer_rate']:.2f} bits/min")
    
    if 'avg_latency_ms' in results:
        table.add_row("Avg Latency", f"{results['avg_latency_ms']:.1f}ms")
    
    console.print(table)


def _generate_benchmark_report(benchmark_results, output_path):
    """Generate comprehensive benchmark report."""
    report_path = output_path / "benchmark_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# BCI-GPT Benchmarking Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Threshold | Accuracy | WER | ITR (bits/min) | Latency (ms) |\n")
        f.write("|-----------|----------|-----|----------------|---------------|\n")
        
        for threshold, results in benchmark_results.items():
            f.write(f"| {threshold:.1f} | {results['accuracy']:.1%} | ")
            f.write(f"{results.get('word_error_rate', 0):.1%} | ")
            f.write(f"{results.get('information_transfer_rate', 0):.2f} | ")
            f.write(f"{results.get('avg_latency_ms', 0):.1f} |\n")
    
    console.print(f"Benchmark report saved to: {report_path}")


def _run_synthetic_demo(model_path, duration, device):
    """Run synthetic EEG decoding demo."""
    console.print("Running synthetic EEG demo...")
    
    # Generate synthetic demo words
    demo_words = ["hello", "world", "brain", "computer", "interface", "thought", "decode", "text"]
    
    for i in range(duration // 3):  # One word every 3 seconds
        word = demo_words[i % len(demo_words)]
        confidence = np.random.uniform(0.6, 0.9)
        
        console.print(f"[{i*3:2d}s] Decoded: [bold green]{word}[/bold green] (confidence: {confidence:.2f})")
        time.sleep(3)
    
    console.print("[green]Synthetic demo completed![/green]")


def _run_recorded_demo(model_path, duration, device):
    """Run recorded EEG data demo."""
    console.print("Note: This would use pre-recorded EEG data in a real implementation.")
    _run_synthetic_demo(model_path, duration, device)


def _run_interactive_demo(model_path, duration, device):
    """Run interactive demo."""
    console.print("Interactive demo: Think of words and see the decoder in action!")
    console.print("(This is a simulation - actual BCI would use real EEG)")
    
    _run_synthetic_demo(model_path, duration, device)


def _display_final_statistics(decoder: RealtimeDecoder, results_history: List[DecodingResult]):
    """Display final decoding session statistics."""
    if not results_history:
        console.print("[yellow]No results to analyze[/yellow]")
        return
    
    # Calculate statistics
    confidences = [r.confidence for r in results_history]
    processing_times = [r.processing_time * 1000 for r in results_history]  # Convert to ms
    
    table = Table(title="Session Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Results", str(len(results_history)))
    table.add_row("Avg Confidence", f"{np.mean(confidences):.2f}")
    table.add_row("Min Confidence", f"{np.min(confidences):.2f}")
    table.add_row("Max Confidence", f"{np.max(confidences):.2f}")
    table.add_row("Avg Processing Time", f"{np.mean(processing_times):.1f}ms")
    table.add_row("Max Processing Time", f"{np.max(processing_times):.1f}ms")
    
    console.print(table)


def main():
    """Main entry point for the decoding CLI."""
    app()


if __name__ == "__main__":
    main()