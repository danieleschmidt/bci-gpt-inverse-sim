# BCI-GPT Tutorial: From Setup to Thought-to-Text

## Tutorial Overview

This hands-on tutorial will guide you through setting up BCI-GPT and decoding your first thoughts to text. By the end, you'll have a working brain-computer interface system.

**Time Required**: 2-3 hours  
**Difficulty**: Intermediate  
**Prerequisites**: Basic Python knowledge, EEG device

## What You'll Learn

1. Set up BCI-GPT environment
2. Connect and configure EEG device
3. Perform user calibration
4. Decode thoughts in real-time
5. Optimize performance
6. Build a simple BCI application

## Part 1: Environment Setup (30 minutes)

### Step 1: Install BCI-GPT

```bash
# Create project directory
mkdir my-bci-project
cd my-bci-project

# Create virtual environment
python -m venv bci-env
source bci-env/bin/activate  # Windows: bci-env\Scripts\activate

# Install BCI-GPT
pip install bci-gpt

# Verify installation
python -c "import bci_gpt; print('BCI-GPT installed successfully!')"
```

### Step 2: Hardware Check

```python
# test_hardware.py
from bci_gpt import DeviceManager

# Check available devices
device_manager = DeviceManager()
devices = device_manager.scan_devices()

print("Available EEG devices:")
for device in devices:
    print(f"- {device.name} ({device.type})")

if not devices:
    print("No EEG devices found. Using simulation mode.")
```

### Step 3: Basic Configuration

Create `config.yaml`:

```yaml
# BCI-GPT Configuration
device:
  type: "openbci"  # or "simulation" for testing
  port: "/dev/ttyUSB0"  # adjust for your system
  channels: ["Fz", "Cz", "Pz", "F3", "F4", "C3", "C4", "P3", "P4"]
  sampling_rate: 1000

processing:
  bandpass_filter: [0.5, 40]
  notch_filter: 60
  buffer_size: 1000

decoding:
  model: "bci-gpt-base"
  confidence_threshold: 0.7
  update_interval: 0.1
```

## Part 2: Device Connection (45 minutes)

### Step 4: Connect EEG Device

```python
# connect_device.py
from bci_gpt import BCISystem
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize BCI system
bci = BCISystem(config)

# Connect to device
try:
    bci.connect()
    print("✅ Device connected successfully!")
    
    # Check signal quality
    quality = bci.check_signal_quality()
    print(f"Signal quality: {quality}")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("💡 Try simulation mode for testing")
```

### Step 5: Signal Quality Check

```python
# signal_check.py
import matplotlib.pyplot as plt
from bci_gpt import SignalMonitor

# Start signal monitoring
monitor = SignalMonitor(bci)
monitor.start()

# Record 10 seconds of data
print("Recording 10 seconds... Keep still and relaxed.")
data = monitor.record(duration=10)

# Visualize signals
fig, axes = plt.subplots(len(config['device']['channels']), 1, figsize=(12, 8))
for i, channel in enumerate(config['device']['channels']):
    axes[i].plot(data[i, :1000])  # Plot first second
    axes[i].set_title(f'Channel {channel}')
    axes[i].set_ylabel('µV')

plt.xlabel('Samples')
plt.tight_layout()
plt.savefig('signal_quality.png')
print("📊 Signal plot saved as 'signal_quality.png'")

# Check for common issues
quality_report = monitor.analyze_quality(data)
if quality_report['overall_score'] < 0.7:
    print("⚠️ Signal quality issues detected:")
    for issue in quality_report['issues']:
        print(f"  - {issue}")
```

## Part 3: User Calibration (60 minutes)

### Step 6: Prepare Calibration Data

```python
# calibration.py
from bci_gpt import CalibrationManager
import time

# Define calibration words
calibration_words = [
    'yes', 'no', 'hello', 'stop', 'help',
    'up', 'down', 'left', 'right', 'select'
]

# Initialize calibration
calibrator = CalibrationManager(bci)

print("🧠 Starting calibration process...")
print("Instructions:")
print("1. Think each word clearly when prompted")
print("2. Avoid movement and muscle tension")
print("3. Focus on 'saying' the word in your mind")
print("4. Take breaks between words if needed")

input("Press Enter when ready to start...")
```

### Step 7: Run Calibration

```python
# Run calibration for each word
calibration_data = {}

for word in calibration_words:
    print(f"
📝 Calibrating word: '{word.upper()}'")
    print("You will be prompted 10 times to think this word.")
    
    word_data = []
    for trial in range(10):
        print(f"Trial {trial + 1}/10")
        print(f"Think: '{word}' (starting in 3 seconds)")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        print("🧠 THINK NOW!")
        
        # Record 2 seconds of thought
        trial_data = calibrator.record_trial(duration=2.0)
        word_data.append(trial_data)
        
        print("✅ Trial complete")
        time.sleep(1)  # Brief pause
    
    calibration_data[word] = word_data
    print(f"✅ Calibration for '{word}' complete")
    
    # Short break between words
    if word != calibration_words[-1]:
        print("Take a 30-second break...")
        time.sleep(30)

print("🎉 Calibration complete!")
```

### Step 8: Train Personal Model

```python
# train_model.py
from bci_gpt import PersonalModelTrainer

# Initialize trainer
trainer = PersonalModelTrainer(
    base_model='bci-gpt-base',
    user_data=calibration_data
)

print("🤖 Training your personal BCI model...")

# Train model (this may take 10-15 minutes)
personal_model = trainer.train(
    epochs=50,
    validation_split=0.2,
    early_stopping=True
)

# Evaluate model
accuracy = trainer.evaluate(personal_model)
print(f"📊 Model accuracy: {accuracy:.1%}")

# Save personal model
personal_model.save('my_bci_model.pt')
print("💾 Personal model saved!")

if accuracy < 0.7:
    print("⚠️ Accuracy is low. Consider:")
    print("  - Recording more calibration data")
    print("  - Improving signal quality")
    print("  - Using more electrodes")
```

## Part 4: Real-time Decoding (30 minutes)

### Step 9: Live Thought-to-Text

```python
# live_decoding.py
from bci_gpt import RealTimeDecoder
import queue
import threading

# Load your personal model
bci.load_model('my_bci_model.pt')

# Initialize decoder
decoder = RealTimeDecoder(
    bci_system=bci,
    confidence_threshold=0.7,
    update_interval=0.5  # Decode every 500ms
)

# Text output queue
text_queue = queue.Queue()

def decode_thoughts():
    """Background thread for decoding."""
    decoder.start()
    
    while True:
        result = decoder.get_next_prediction()
        if result and result.confidence > 0.7:
            text_queue.put(result.text)

# Start decoding thread
decode_thread = threading.Thread(target=decode_thoughts)
decode_thread.daemon = True
decode_thread.start()

print("🧠 Live decoding started!")
print("Think one of your calibrated words...")
print("Press Ctrl+C to stop")

# Main loop
try:
    while True:
        try:
            # Get decoded text (non-blocking)
            text = text_queue.get_nowait()
            print(f"🗣️ Decoded: '{text}'")
        except queue.Empty:
            time.sleep(0.1)
            
except KeyboardInterrupt:
    print("
👋 Stopping decoder...")
    decoder.stop()
```

### Step 10: Build Simple BCI App

```python
# bci_app.py
import tkinter as tk
from tkinter import scrolledtext
import threading

class BCITextApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BCI Text Interface")
        self.root.geometry("600x400")
        
        # Setup UI
        self.setup_ui()
        
        # Setup BCI
        self.setup_bci()
        
    def setup_ui(self):
        # Text display
        self.text_area = scrolledtext.ScrolledText(
            self.root, 
            height=15, 
            width=70,
            font=("Arial", 12)
        )
        self.text_area.pack(pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = tk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Control buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)
        
        tk.Button(
            button_frame,
            text="Start Decoding",
            command=self.start_decoding
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame,
            text="Stop Decoding",
            command=self.stop_decoding
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame,
            text="Clear Text",
            command=self.clear_text
        ).pack(side=tk.LEFT, padx=5)
    
    def setup_bci(self):
        # Initialize BCI system
        self.bci = BCISystem(config)
        self.bci.connect()
        self.bci.load_model('my_bci_model.pt')
        
        self.decoder = RealTimeDecoder(self.bci)
        self.decoding = False
    
    def start_decoding(self):
        if not self.decoding:
            self.decoding = True
            self.status_var.set("Decoding active...")
            
            # Start decoding in background
            self.decode_thread = threading.Thread(target=self.decode_loop)
            self.decode_thread.daemon = True
            self.decode_thread.start()
    
    def stop_decoding(self):
        self.decoding = False
        self.status_var.set("Decoding stopped")
    
    def decode_loop(self):
        self.decoder.start()
        
        while self.decoding:
            result = self.decoder.get_next_prediction()
            if result and result.confidence > 0.7:
                # Update UI in main thread
                self.root.after(0, self.add_text, result.text)
    
    def add_text(self, text):
        self.text_area.insert(tk.END, f"{text} ")
        self.text_area.see(tk.END)
    
    def clear_text(self):
        self.text_area.delete(1.0, tk.END)
    
    def run(self):
        self.root.mainloop()

# Run the app
if __name__ == "__main__":
    app = BCITextApp()
    app.run()
```

## Part 5: Performance Optimization (15 minutes)

### Step 11: Measure Performance

```python
# performance_test.py
from bci_gpt import PerformanceAnalyzer
import time

analyzer = PerformanceAnalyzer(bci)

# Test decoding speed
print("🚀 Testing decoding performance...")

start_time = time.time()
for i in range(100):
    # Simulate real-time decoding
    fake_eeg = generate_test_signal()
    result = decoder.decode(fake_eeg)

end_time = time.time()
avg_latency = (end_time - start_time) / 100

print(f"Average decoding latency: {avg_latency*1000:.1f}ms")

# Test accuracy on calibration data
accuracy = analyzer.test_accuracy(calibration_data)
print(f"Overall accuracy: {accuracy:.1%}")

# Generate performance report
report = analyzer.generate_report()
print("
📊 Performance Report:")
print(f"  Signal Quality: {report['signal_quality']:.1%}")
print(f"  Decoding Speed: {report['decoding_speed']:.1f} Hz")
print(f"  Memory Usage: {report['memory_mb']:.1f} MB")
```

### Step 12: Optimization Tips

```python
# optimization.py

# 1. Optimize buffer size
bci.set_buffer_size(500)  # Reduce for lower latency

# 2. Use GPU acceleration (if available)
bci.enable_gpu()

# 3. Adjust confidence threshold
decoder.set_confidence_threshold(0.6)  # Lower for more responsive

# 4. Enable preprocessing cache
bci.enable_preprocessing_cache(True)

# 5. Use quantized model for speed
bci.load_model('my_bci_model_quantized.pt')
```

## Troubleshooting

### Common Issues and Solutions

1. **"Device not found"**
   ```bash
   # Check USB connections
   lsusb
   
   # Check permissions
   sudo chmod 666 /dev/ttyUSB0
   ```

2. **"Poor signal quality"**
   - Check electrode contact
   - Apply conductive gel
   - Reduce muscle tension
   - Move away from electrical interference

3. **"Low accuracy"**
   - Record more calibration data
   - Use more electrodes
   - Improve signal quality
   - Try different words

4. **"High latency"**
   - Reduce buffer size
   - Enable GPU acceleration
   - Close other applications
   - Use quantized model

## Next Steps

Congratulations! You've built a working BCI system. Here are some ideas for further exploration:

### Beginner Projects
- Add more vocabulary words
- Build a BCI-controlled game
- Create a BCI typing interface

### Intermediate Projects  
- Train on continuous speech
- Add multiple languages
- Build a BCI web app

### Advanced Projects
- Research novel architectures
- Contribute to open source
- Publish your findings

## Resources

- **Documentation**: https://docs.bci-gpt.com
- **Examples**: https://github.com/danieleschmidt/bci-gpt-examples
- **Community**: https://discord.gg/bci-gpt
- **Datasets**: https://datasets.bci-gpt.com

---

**Congratulations on completing the BCI-GPT tutorial!** 🎉

You now have the skills to build brain-computer interfaces and decode thoughts to text. Keep experimenting and building amazing BCI applications!
