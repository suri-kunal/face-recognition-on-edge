# Face Recognition on Edge

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive face recognition pipeline designed for edge computing, utilizing Google's MediaPipe for face detection and ArcFace for feature extraction. This project implements a complete pipeline for recognizing faces in group photos using state-of-the-art computer vision techniques.

## Features

- **Face Detection**: Uses Google's MediaPipe for efficient and accurate face detection
- **Feature Extraction**: Leverages ArcFace model for high-quality 512-dimensional face embeddings
- **Face Matching**: Implements cosine similarity-based matching with configurable thresholds
- **LFW Dataset Support**: Built-in utilities for downloading and working with the Labeled Faces in the Wild dataset
- **Batch Processing**: Support for processing multiple images efficiently
- **Evaluation Tools**: Comprehensive evaluation scripts with ROC curves and accuracy metrics

## Pipeline Overview

1. **Face Detection**: Find faces in group photos using MediaPipe
2. **Feature Extraction**: Convert face images to 512-dimensional vectors using ArcFace
3. **Comparison**: Measure similarity between probe and candidate faces using cosine similarity  
4. **Threshold Decision**: Determine matches based on configurable similarity threshold

## Installation

### Prerequisites
- Python 3.12 or higher
- Virtual environment (recommended)

### Setup Instructions

```bash
# Clone the repository
git clone <repository-url>
cd face-recognition-on-edge

# Create and activate virtual environment
uv sync
source .venv/bin/activate

# Install the package in editable mode
pip install -e .
```

This installs the package in development mode, allowing you to make changes to the source code while using the package.

## Quick Start

```python
from face_recognition_on_edge import FaceRecognitionPipeline, LFWDataset

# Initialize pipeline
pipeline = FaceRecognitionPipeline(
    similarity_threshold=0.5,
    detection_confidence=0.5
)

# Recognize face in group photo
results = pipeline.recognize_face(
    probe_image_path="path/to/individual.jpg",
    group_photo_path="path/to/group.jpg"
)

print(f"Match found: {results['is_match']}")
print(f"Similarity score: {results['best_similarity']:.3f}")
```

## Usage Examples

### Basic Face Recognition

```python
from face_recognition_on_edge import FaceRecognitionPipeline

pipeline = FaceRecognitionPipeline()
results = pipeline.recognize_face("probe.jpg", "group.jpg")

if results['success']:
    print(f"Best match index: {results['best_match_index']}")
    print(f"Similarity: {results['best_similarity']:.3f}")
    print(f"Is match: {results['is_match']}")
```

### Batch Recognition

```python
probe_images = ["person1.jpg", "person2.jpg", "person3.jpg"]
results = pipeline.batch_recognize(probe_images, "group.jpg")

for i, result in enumerate(results):
    print(f"Person {i+1}: Match = {result['is_match']}")
```

### Working with LFW Dataset

```python
from face_recognition_on_edge import LFWDataset

# Setup LFW dataset
lfw = LFWDataset("data/lfw")
lfw.setup_dataset()  # Downloads and extracts LFW data

# Create simulated group photo
group_photo, individuals = lfw.create_group_photo_simulation(num_faces=5)

# Use in pipeline
results = pipeline.recognize_face(individuals[0], group_photo)
```

## Running Examples

### Basic Demo
```bash
python examples/demo.py
```

### Evaluation
```bash
python examples/evaluate_pipeline.py
```

This will:
- Download the LFW dataset
- Evaluate the pipeline on face pairs
- Generate ROC curves
- Test group photo recognition with different group sizes

## Project Structure

```

