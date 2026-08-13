# 🛡️ VisionGuard

### Privacy-Aware AI Content Moderation System for Images and Videos

VisionGuard is an AI-powered content moderation system designed for smart glasses, wearable cameras, edge AI devices, surveillance systems, and privacy-sensitive applications.

The system analyzes images and videos in real time and classifies content as:

- ✅ ALLOW
- ⚠️ UNCERTAIN
- 🚫 BLOCK

Unlike traditional moderation systems that rely on a single NSFW classifier, VisionGuard uses a multi-model ensemble approach combined with person detection, image enhancement, and explicit nudity verification.

---

# 🚀 Features

## Image Moderation

- Image Quality Analysis
- Blur Detection
- Brightness Analysis
- Resolution Validation
- Conditional Enhancement
- YOLOv8 Person Detection
- Adaptive Person Cropping
- Multi-Model Content Analysis
- Explicit Nudity Detection

## Video Moderation

- Intelligent Frame Sampling
- Video Frame Enhancement
- Batch YOLO Inference
- Person Detection & Cropping
- Multi-Model Moderation
- Early Violation Detection
- Immediate Pipeline Termination
- Evidence Generation

## Privacy-Centric Design

- Only detected person regions are analyzed
- Full images are never directly moderated
- Explicit-content decisions require object-level evidence
- Reduced false positives compared to traditional NSFW systems

---

# 🏗️ System Architecture

## Image Pipeline

```text
Image
 │
 ▼
Quality Analysis
 │
 ▼
Conditional Enhancement
 │
 ▼
YOLOv8 Person Detection
 │
 ▼
Adaptive Person Crops
 │
 ▼
FalconS-AI
 │
 ▼
Freepik NSFW
 │
 ▼
NudeNet
 │
 ▼
Decision Engine
 │
 ├── BLOCK
 ├── UNCERTAIN
 └── ALLOW
```

## Video Pipeline

```text
Video
 │
 ▼
Frame Sampler (2 FPS)
 │
 ▼
Batch Processor (16 Frames)
 │
 ▼
Video Frame Enhancement
 │
 ▼
YOLOv8 Batch Detection
 │
 ▼
Person Crops
 │
 ▼
FalconS-AI
 │
 ▼
Freepik NSFW
 │
 ▼
NudeNet
 │
 ▼
Decision Engine
 │
 ├── BLOCK
 ├── UNCERTAIN
 └── ALLOW
```

---

# 🎥 Video Enhancement

Before moderation, every sampled frame is enhanced to improve detection quality.

### CLAHE

Improves:

- Low-light scenes
- Indoor recordings
- Shadow regions
- Local contrast

### Unsharp Masking

Improves:

- Mild motion blur
- Soft edges
- Subject visibility

### Benefits

- Better person detection
- Better crop quality
- Better moderation accuracy
- Improved detection under challenging conditions

---

# 🤖 AI Models Used

## YOLOv8m

**Purpose:** Person Detection

- Model: YOLOv8 Medium
- Confidence Threshold: 0.25
- Input Size: 640×640
- Batch Inference Support

---

## FalconS-AI

**Model**

```text
Falconsai/nsfw_image_detection
```

**Purpose**

Binary NSFW classification.

Output:

```python
{
    "nsfw_score": float,
    "safe_score": float,
    "predicted_class": str
}
```

Role:

```text
Risk Signal Only
```

---

## Freepik NSFW Detector

Provides multi-level NSFW risk estimation.

Levels:

```text
neutral
low
medium
high
```

Role:

```text
Risk Signal Only
```

---

## NudeNet

Object-level explicit body-part detection.

Dangerous Classes:

```text
FEMALE_BREAST_EXPOSED
FEMALE_GENITALIA_EXPOSED
MALE_GENITALIA_EXPOSED
BUTTOCKS_EXPOSED
ANUS_EXPOSED
```

Role:

```text
Only model allowed to BLOCK content
```

---

# 📋 Decision Policy

| Decision | Condition |
|-----------|------------|
| 🚫 BLOCK | NudeNet confirms explicit nudity |
| ⚠️ UNCERTAIN | FalconS-AI or Freepik detects suspicious content |
| ✅ ALLOW | No explicit or suspicious content detected |

### Important

NudeNet is the **only model** that can issue a BLOCK decision.

FalconS-AI and Freepik act as supporting risk indicators and generate UNCERTAIN results instead of hard blocks.

This significantly reduces false positives.

---

# ⚡ Performance Optimizations

### Batch YOLO Inference

Instead of:

```text
1 YOLO inference per frame
```

VisionGuard performs:

```text
1 YOLO inference per 16 frames
```

Benefits:

- Faster processing
- Better GPU utilization
- Reduced latency

---

### Early Termination

When a violation is detected:

```text
Stop remaining persons
Stop remaining frames
Stop remaining batches
Return immediately
```

Benefits:

- Faster moderation
- Lower compute cost
- Real-time suitability

---

# 📂 Project Structure

```text
VisionGuard/
│
├── main.py
├── requirements.txt
├── yolov8m.pt
│
├── src/
│   ├── models/
│   ├── pipeline/
│   ├── preprocessing/
│   ├── video/
│   └── utils/
│
├── tests/
│
└── outputs/
    ├── flagged_frames/
    ├── flagged_crops/
    ├── debug/
    └── allowed/
```

---

# 🖥️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/VisionGuard.git
cd VisionGuard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Additional packages:

```bash
pip install torch ultralytics nudenet nsfw_image_detector
```

---

# ▶️ Run

```bash
python main.py
```

Select an image or video file using the file picker.

The system will automatically run the appropriate moderation pipeline.

---

# 📊 Outputs

```text
outputs/
│
├── flagged_frames/
├── flagged_crops/
├── debug/
└── allowed/
```

### flagged_frames

Stores the frame responsible for a BLOCK decision.

### flagged_crops

Stores the exact person crop that triggered the violation.

### debug

Stores suspicious crops for false-positive analysis and threshold tuning.

---

# 🛠️ Tech Stack

- Python
- PyTorch
- Transformers
- OpenCV
- Pillow
- NumPy
- Ultralytics YOLOv8
- FalconS-AI
- Freepik NSFW
- NudeNet

---

# 🔮 Future Improvements

- Person Tracking Across Frames
- Temporal Consistency Analysis
- Audio Moderation
- Live Camera Support
- RTSP Stream Moderation
- Human Review Dashboard
- Active Learning Feedback Loop
- Custom VisionGuard Dataset
- Ensemble Confidence Calibration

---

# 👨‍💻 Author

### Kishore M

Machine Learning Engineer | Full-Stack Developer

VisionGuard is a privacy-first AI moderation engine designed for real-world deployment in smart devices, wearable cameras, edge AI systems, and content safety applications.
