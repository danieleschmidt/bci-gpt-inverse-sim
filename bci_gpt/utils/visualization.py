"""Visualization tools for EEG data and BCI-GPT results."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available for visualization")

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import plot
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available for interactive visualization")


class EEGVisualizer:
    """Comprehensive visualization tools for EEG data and BCI results."""
    
    def __init__(self, style: str = "seaborn", figsize: Tuple[int, int] = (12, 8)):
        """Initialize EEG visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.figsize = figsize
        
        if HAS_MATPLOTLIB:
            try:
                plt.style.use(style)
            except:
                pass  # Use default style if requested style not available
            
            # Set seaborn palette if available
            if HAS_MATPLOTLIB and 'seaborn' in str(plt.style.available):
                try:
                    sns.set_palette("husl")
                except:
                    pass
    
    def plot_eeg_signals(self,
                        eeg_data: np.ndarray,
                        channels: Optional[List[str]] = None,
                        sampling_rate: int = 1000,
                        time_range: Optional[Tuple[float, float]] = None,
                        amplitude_scale: float = 1.0,
                        show_grid: bool = True,
                        save_path: Optional[str] = None) -> None:
        """Plot multi-channel EEG signals.
        
        Args:
            eeg_data: EEG data (channels x samples)
            channels: Channel names
            sampling_rate: Sampling rate in Hz
            time_range: Time range to plot (start, end) in seconds
            amplitude_scale: Scaling factor for amplitudes
            show_grid: Whether to show grid
            save_path: Optional path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return
        
        n_channels, n_samples = eeg_data.shape
        
        # Create time axis
        time_axis = np.arange(n_samples) / sampling_rate
        
        # Apply time range filter
        if time_range:
            start_idx = int(time_range[0] * sampling_rate)
            end_idx = int(time_range[1] * sampling_rate)
            time_axis = time_axis[start_idx:end_idx]
            eeg_data = eeg_data[:, start_idx:end_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Channel names
        if channels is None:
            channels = [f'Ch{i+1}' for i in range(n_channels)]
        
        # Plot signals with offset
        offset_step = np.max(np.std(eeg_data, axis=1)) * 4 * amplitude_scale
        
        for ch_idx in range(n_channels):
            signal = eeg_data[ch_idx] * amplitude_scale
            offset = ch_idx * offset_step
            
            ax.plot(time_axis, signal + offset, 
                   label=channels[ch_idx], linewidth=0.8)
            
            # Add channel labels
            ax.text(-0.02, offset, channels[ch_idx], 
                   transform=ax.get_yaxis_transform(),
                   fontsize=10, ha='right', va='center')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channels')
        ax.set_title('Multi-Channel EEG Signals')
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        # Remove y-axis ticks (not meaningful with offset)
        ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_spectrogram(self,
                        eeg_data: np.ndarray,
                        channel_idx: int = 0,
                        sampling_rate: int = 1000,
                        nperseg: int = 256,
                        noverlap: Optional[int] = None,
                        freq_range: Tuple[float, float] = (0.5, 50),
                        save_path: Optional[str] = None) -> None:
        """Plot spectrogram of EEG signal.
        
        Args:
            eeg_data: EEG data (channels x samples)
            channel_idx: Channel index to plot
            sampling_rate: Sampling rate in Hz
            nperseg: Length of each segment for FFT
            noverlap: Number of points to overlap between segments
            freq_range: Frequency range to display
            save_path: Optional path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return
        
        try:
            from scipy import signal
        except ImportError:
            print("SciPy required for spectrogram")
            return
        
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Compute spectrogram
        freqs, times, Sxx = signal.spectrogram(
            eeg_data[channel_idx], 
            fs=sampling_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        # Filter frequency range
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[freq_mask]
        Sxx = Sxx[freq_mask, :]
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.pcolormesh(times, freqs, 10 * np.log10(Sxx), 
                          shading='gouraud', cmap='viridis')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Spectrogram - Channel {channel_idx + 1}')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_power_spectral_density(self,
                                   eeg_data: np.ndarray,
                                   channels: Optional[List[str]] = None,
                                   sampling_rate: int = 1000,
                                   freq_range: Tuple[float, float] = (0.5, 100),
                                   log_scale: bool = True,
                                   show_bands: bool = True,
                                   save_path: Optional[str] = None) -> None:
        """Plot power spectral density for all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            channels: Channel names
            sampling_rate: Sampling rate in Hz
            freq_range: Frequency range to display
            log_scale: Whether to use log scale for power
            show_bands: Whether to show EEG frequency bands
            save_path: Optional path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return
        
        try:
            from scipy import signal
        except ImportError:
            print("SciPy required for PSD calculation")
            return
        
        n_channels = eeg_data.shape[0]
        
        if channels is None:
            channels = [f'Ch{i+1}' for i in range(n_channels)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # EEG frequency bands
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 100)
        }
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
        
        for ch_idx in range(n_channels):
            # Compute PSD
            freqs, psd = signal.welch(
                eeg_data[ch_idx], 
                fs=sampling_rate,
                nperseg=min(1024, eeg_data.shape[1]//4)
            )
            
            # Filter frequency range
            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            freqs_filtered = freqs[freq_mask]
            psd_filtered = psd[freq_mask]
            
            # Plot PSD
            if log_scale:
                ax.semilogy(freqs_filtered, psd_filtered, 
                           color=colors[ch_idx], label=channels[ch_idx], alpha=0.8)
            else:
                ax.plot(freqs_filtered, psd_filtered, 
                       color=colors[ch_idx], label=channels[ch_idx], alpha=0.8)
        
        # Show frequency bands
        if show_bands:
            y_min, y_max = ax.get_ylim()
            for band_name, (low, high) in bands.items():
                if low >= freq_range[0] and high <= freq_range[1]:
                    ax.axvspan(low, high, alpha=0.1, label=f'{band_name} ({low}-{high} Hz)')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title('Power Spectral Density - All Channels')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_topographic_map(self,
                           values: np.ndarray,
                           channels: List[str],
                           title: str = "Topographic Map",
                           cmap: str = "RdBu_r",
                           save_path: Optional[str] = None) -> None:
        """Plot topographic map of EEG values.
        
        Args:
            values: Values for each channel
            channels: Channel names (must be standard 10-20 names)
            title: Plot title
            cmap: Colormap name
            save_path: Optional path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return
        
        # Standard 10-20 electrode positions (simplified)
        electrode_positions = {
            'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
            'F7': (-0.7, 0.4), 'F3': (-0.4, 0.4), 'Fz': (0, 0.4), 'F4': (0.4, 0.4), 'F8': (0.7, 0.4),
            'T7': (-0.9, 0), 'C3': (-0.4, 0), 'Cz': (0, 0), 'C4': (0.4, 0), 'T8': (0.9, 0),
            'P7': (-0.7, -0.4), 'P3': (-0.4, -0.4), 'Pz': (0, -0.4), 'P4': (0.4, -0.4), 'P8': (0.7, -0.4),
            'O1': (-0.3, -0.9), 'O2': (0.3, -0.9)
        }
        
        # Filter channels that have known positions
        valid_channels = [ch for ch in channels if ch in electrode_positions]
        if not valid_channels:
            print("No valid 10-20 channel names found for topographic mapping")
            return
        
        # Get positions and values for valid channels
        positions = np.array([electrode_positions[ch] for ch in valid_channels])
        channel_values = np.array([values[channels.index(ch)] for ch in valid_channels])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create interpolated surface
        from scipy.interpolate import griddata
        
        # Create grid
        xi = np.linspace(-1, 1, 100)
        yi = np.linspace(-1, 1, 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate values
        zi = griddata(positions, channel_values, (xi, yi), method='cubic')
        
        # Create circular mask
        mask = (xi**2 + yi**2) <= 1
        zi[~mask] = np.nan
        
        # Plot interpolated surface
        im = ax.contourf(xi, yi, zi, levels=20, cmap=cmap, extend='both')
        
        # Plot electrode positions
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                           c=channel_values, s=100, cmap=cmap, 
                           edgecolors='black', linewidth=2, zorder=5)
        
        # Add channel labels
        for i, ch in enumerate(valid_channels):
            ax.annotate(ch, (positions[i, 0], positions[i, 1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold')
        
        # Draw head outline
        head_circle = plt.Circle((0, 0), 1, fill=False, linewidth=3, color='black')
        ax.add_patch(head_circle)
        
        # Draw nose
        nose = patches.Polygon([(-0.1, 1), (0, 1.1), (0.1, 1)], 
                             closed=True, fill=False, linewidth=2, color='black')
        ax.add_patch(nose)
        
        # Draw ears
        left_ear = patches.Arc((-1, 0), 0.2, 0.4, angle=90, 
                              theta1=-30, theta2=30, linewidth=2, color='black')
        right_ear = patches.Arc((1, 0), 0.2, 0.4, angle=90, 
                               theta1=150, theta2=210, linewidth=2, color='black')
        ax.add_patch(left_ear)
        ax.add_patch(right_ear)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Amplitude', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_decoding_results(self,
                            predicted_texts: List[str],
                            reference_texts: List[str],
                            confidences: List[float],
                            save_path: Optional[str] = None) -> None:
        """Plot decoding results with confidence scores.
        
        Args:
            predicted_texts: Predicted text strings
            reference_texts: Reference text strings
            confidences: Confidence scores
            save_path: Optional path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return
        
        n_samples = len(predicted_texts)
        
        # Calculate accuracy for each sample
        accuracies = [pred == ref for pred, ref in zip(predicted_texts, reference_texts)]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot confidence scores
        colors = ['green' if acc else 'red' for acc in accuracies]
        bars = ax1.bar(range(n_samples), confidences, color=colors, alpha=0.7)
        
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Confidence Score')
        ax1.set_title('Decoding Confidence Scores (Green=Correct, Red=Incorrect)')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add threshold line
        threshold = 0.7  # Example threshold
        ax1.axhline(y=threshold, color='orange', linestyle='--', 
                   label=f'Threshold ({threshold:.1f})')
        ax1.legend()
        
        # Plot text comparison (show first 10 samples)
        show_samples = min(10, n_samples)
        y_pos = np.arange(show_samples)
        
        # Prepare text display
        pred_display = [pred[:30] + '...' if len(pred) > 30 else pred 
                       for pred in predicted_texts[:show_samples]]
        ref_display = [ref[:30] + '...' if len(ref) > 30 else ref 
                      for ref in reference_texts[:show_samples]]
        
        ax2.barh(y_pos - 0.2, [1] * show_samples, height=0.4, 
                color='lightblue', alpha=0.7, label='Predicted')
        ax2.barh(y_pos + 0.2, [1] * show_samples, height=0.4, 
                color='lightgreen', alpha=0.7, label='Reference')
        
        # Add text labels
        for i in range(show_samples):
            ax2.text(0.02, i - 0.2, pred_display[i], va='center', fontsize=8)
            ax2.text(0.02, i + 0.2, ref_display[i], va='center', fontsize=8)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f'Sample {i}' for i in range(show_samples)])
        ax2.set_xlabel('Text')
        ax2.set_title('Text Comparison (First 10 Samples)')
        ax2.legend()
        ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_curves(self,
                           history: Dict[str, List[float]],
                           save_path: Optional[str] = None) -> None:
        """Plot training curves from training history.
        
        Args:
            history: Training history dictionary
            save_path: Optional path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return
        
        # Determine subplot layout
        metrics = list(history.keys())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            print("No metrics to plot")
            return
        
        cols = 2 if n_metrics > 1 else 1
        rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = history[metric]
            epochs = range(1, len(values) + 1)
            
            axes[i].plot(epochs, values, 'b-', linewidth=2, label=f'Train {metric}')
            
            # If validation metric exists, plot it too
            val_metric = f'val_{metric}'
            if val_metric in history:
                val_values = history[val_metric]
                val_epochs = range(1, len(val_values) + 1)
                axes[i].plot(val_epochs, val_values, 'r--', linewidth=2, 
                           label=f'Validation {metric}')
            
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'Training {metric.capitalize()}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_eeg_plot(self,
                                   eeg_data: np.ndarray,
                                   channels: Optional[List[str]] = None,
                                   sampling_rate: int = 1000,
                                   save_html: Optional[str] = None) -> None:
        """Create interactive EEG plot using Plotly.
        
        Args:
            eeg_data: EEG data (channels x samples)
            channels: Channel names
            sampling_rate: Sampling rate in Hz
            save_html: Optional path to save HTML file
        """
        if not HAS_PLOTLY:
            print("Plotly not available for interactive plotting")
            return
        
        n_channels, n_samples = eeg_data.shape
        
        if channels is None:
            channels = [f'Ch{i+1}' for i in range(n_channels)]
        
        # Create time axis
        time_axis = np.arange(n_samples) / sampling_rate
        
        # Create subplots
        fig = sp.make_subplots(
            rows=n_channels, cols=1,
            subplot_titles=channels,
            shared_xaxes=True,
            vertical_spacing=0.02
        )
        
        # Add traces for each channel
        for ch_idx in range(n_channels):
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=eeg_data[ch_idx],
                    mode='lines',
                    name=channels[ch_idx],
                    line=dict(width=1),
                    showlegend=False
                ),
                row=ch_idx + 1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Interactive EEG Signals",
            xaxis_title="Time (s)",
            height=150 * n_channels,
            showlegend=False
        )
        
        # Update x-axis for bottom subplot only
        fig.update_xaxes(title_text="Time (s)", row=n_channels, col=1)
        
        if save_html:
            plot(fig, filename=save_html, auto_open=False)
        else:
            fig.show()
    
    def animate_real_time_eeg(self,
                            data_generator,
                            channels: Optional[List[str]] = None,
                            window_duration: float = 5.0,
                            sampling_rate: int = 1000,
                            update_interval: int = 100) -> None:
        """Create animated real-time EEG plot.
        
        Args:
            data_generator: Generator that yields new EEG data chunks
            channels: Channel names
            window_duration: Duration of sliding window in seconds
            sampling_rate: Sampling rate in Hz
            update_interval: Animation update interval in ms
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for animation")
            return
        
        # Initialize plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        window_samples = int(window_duration * sampling_rate)
        n_channels = len(channels) if channels else 9
        
        if channels is None:
            channels = [f'Ch{i+1}' for i in range(n_channels)]
        
        # Initialize data buffer
        data_buffer = np.zeros((n_channels, window_samples))
        time_axis = np.arange(window_samples) / sampling_rate
        
        # Initialize line objects
        lines = []
        offset_step = 100  # μV
        
        for ch_idx in range(n_channels):
            line, = ax.plot(time_axis, data_buffer[ch_idx] + ch_idx * offset_step,
                           label=channels[ch_idx])
            lines.append(line)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channels')
        ax.set_title('Real-Time EEG')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        def animate(frame):
            try:
                # Get new data chunk
                new_chunk = next(data_generator)
                
                if new_chunk is not None and new_chunk.shape[0] == n_channels:
                    # Shift buffer and add new data
                    chunk_size = new_chunk.shape[1]
                    data_buffer[:, :-chunk_size] = data_buffer[:, chunk_size:]
                    data_buffer[:, -chunk_size:] = new_chunk
                    
                    # Update line data
                    for ch_idx, line in enumerate(lines):
                        line.set_ydata(data_buffer[ch_idx] + ch_idx * offset_step)
                
            except StopIteration:
                pass  # No more data
            except Exception as e:
                print(f"Animation error: {e}")
            
            return lines
        
        # Create animation
        anim = FuncAnimation(fig, animate, interval=update_interval, 
                           blit=True, repeat=True)
        
        plt.tight_layout()
        plt.show()
        
        return anim