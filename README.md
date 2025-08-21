# Face Recognition on Edge

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive face recognition pipeline designed for edge computing, featuring multiple recognition backends including DeepFace integration and custom ArcFace implementation. This project provides flexible pipelines for recognizing faces in group photos using state-of-the-art computer vision techniques.

## Features

- **Multiple Recognition Backends**: 
  - **DeepFace Integration**: Support for 9+ models (ArcFace, Facenet, VGG-Face, etc.) with optimized model preloading
  - **Custom ArcFace Pipeline**: Direct ArcFace implementation with 512-dimensional embeddings
- **Face Detection**: Uses Google's MediaPipe for efficient and accurate face detection
- **Model Preloading**: Optimized memory management with model caching for improved performance
- **Flexible Matching**: Multiple distance metrics and configurable thresholds per model
- **LFW Dataset Support**: Built-in utilities for downloading and working with the Labeled Faces in the Wild dataset
- **Batch Processing**: Support for processing multiple images efficiently
- **Evaluation Tools**: Comprehensive evaluation scripts with ROC curves and accuracy metrics

## Pipeline Overview

### DeepFace Pipeline (Recommended)
1. **Face Detection**: Find faces in group photos using MediaPipe
2. **Model Selection**: Choose from multiple DeepFace models (ArcFace, Facenet, VGG-Face, etc.)
3. **Feature Extraction**: Extract embeddings using selected DeepFace models
4. **Comparison**: Measure similarity using model-specific distance metrics
5. **Threshold Decision**: Determine matches based on model-specific thresholds

### Custom ArcFace Pipeline
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

### DeepFace Pipeline (Recommended)

```python
from face_recognition_on_edge.deepface_pipeline import DeepFacePipeline

# Initialize DeepFace pipeline with multiple models
pipeline = DeepFacePipeline(
    models=['ArcFace', 'Facenet', 'VGG-Face'],  # Use multiple models for better accuracy
    detection_confidence=0.5,
    preload_models=True  # Enable model caching for better performance
)

# Recognize face in group photo
results = pipeline.recognize_face(
    probe_image_path="path/to/individual.jpg",
    group_photo_path="path/to/group.jpg"
)

# Print results
pipeline.print_results(results)
```

### Custom ArcFace Pipeline

```python
from face_recognition_on_edge import FaceRecognitionPipeline, LFWDataset

# Initialize custom pipeline
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

### DeepFace Pipeline Examples

#### Basic Recognition with Single Model
```python
from face_recognition_on_edge.deepface_pipeline import DeepFacePipeline

# Initialize with single model
pipeline = DeepFacePipeline(models=['ArcFace'])
results = pipeline.recognize_face("probe.jpg", "group.jpg")

if results['success']:
    print(f"Overall match: {results['overall_match_found']}")
    print(f"Candidates found: {results['num_candidates']}")
    
    # Check results for each model
    for model, result in results['results_by_model'].items():
        if result['match_found']:
            best = result['best_match']
            print(f"{model}: Match found (distance: {best['distance']:.3f})")
```

#### Multi-Model Recognition for Higher Accuracy
```python
# Use multiple models for consensus-based matching
pipeline = DeepFacePipeline(
    models=['ArcFace', 'Facenet', 'VGG-Face'],
    preload_models=True,  # Cache models for better performance
    custom_thresholds={
        'ArcFace': 0.68,
        'Facenet': 0.4,
        'VGG-Face': 0.4
    }
)

results = pipeline.recognize_face("probe.jpg", "group.jpg")
pipeline.print_results(results)  # Pretty-printed results
```

### Custom ArcFace Pipeline Examples

#### Basic Recognition
```python
from face_recognition_on_edge import FaceRecognitionPipeline

pipeline = FaceRecognitionPipeline()
results = pipeline.recognize_face("probe.jpg", "group.jpg")

if results['success']:
    print(f"Best match index: {results['best_match_index']}")
    print(f"Similarity: {results['best_similarity']:.3f}")
    print(f"Is match: {results['is_match']}")
```

#### Batch Recognition
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

### DeepFace Examples

#### Simple DeepFace Recognition
```bash
python examples/simple_deepface_match.py probe.jpg group.jpg
```

#### Optimized Pipeline with Model Preloading
```bash
python examples/test_optimized_pipeline.py
```

#### Memory Profiling (Optional)
```bash
python -m memory_profiler examples/test_optimized_pipeline.py
```

### Custom Pipeline Examples

#### Basic Demo
```bash
python examples/demo.py
```

#### Simple Face Match
```bash
python examples/simple_face_match.py probe.jpg group.jpg
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
face-recognition-on-edge/
├── src/face_recognition_on_edge/
│   ├── __init__.py
│   ├── deepface_pipeline.py          # DeepFace integration pipeline (recommended)
│   ├── face_detector.py              # MediaPipe face detection
│   ├── face_matcher.py               # Custom face matching logic
│   ├── feature_extractor.py          # ArcFace feature extraction
│   ├── lfw_dataset.py                # LFW dataset utilities
│   └── pipeline.py                   # Custom ArcFace pipeline
├── examples/
│   ├── simple_deepface_match.py      # Simple DeepFace example
│   ├── test_optimized_pipeline.py    # Optimized pipeline with preloading
│   ├── simple_face_match.py          # Simple custom pipeline example
│   ├── demo.py                       # Basic demo
│   └── evaluate_pipeline.py          # Evaluation scripts
├── data/lfw/                         # LFW dataset (auto-downloaded)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Supported Models

### DeepFace Models (via DeepFacePipeline)
- **ArcFace** (threshold: 0.68) - Recommended for accuracy
- **Facenet** (threshold: 0.4) - Good balance of speed and accuracy  
- **Facenet512** (threshold: 0.3) - Higher dimensional embeddings
- **VGG-Face** (threshold: 0.4) - Classic model
- **OpenFace** (threshold: 0.1) - Lightweight option
- **DeepFace** (threshold: 0.23) - Facebook's model
- **DeepID** (threshold: 0.015) - Academic model
- **Dlib** (threshold: 0.07) - Traditional approach
- **SFace** (threshold: 0.593) - Recent addition

All models use cosine distance metric and support custom threshold configuration.

## Recent Updates

### v1.1.0 - DeepFace Integration
- ✅ **Fixed DeepFace model preloading**: Resolved `build_model` attribute error
- ✅ **Added DeepFacePipeline**: New pipeline supporting 9+ recognition models
- ✅ **Model caching**: Optimized memory usage with singleton pattern
- ✅ **Multi-model support**: Use multiple models for consensus-based matching
- ✅ **Memory profiling**: Added optional memory profiling support
- ✅ **Custom thresholds**: Per-model threshold configuration

### Bug Fixes
- Fixed `deepface.modules.representation.build_model` error by using correct `modeling.build_model` API
- Added proper memory profiler decorator handling
- Improved error handling and logging

## Performance Considerations

### Model Preloading
- Enable `preload_models=True` for better performance in production
- Models are cached using DeepFace's singleton pattern
- First recognition may be slower due to model loading
- Subsequent recognitions benefit from cached models

### Memory Usage
- DeepFace models can be memory intensive
- Use memory profiling to monitor usage: `python -m memory_profiler script.py`
- Consider using fewer models simultaneously if memory is constrained
- Garbage collection is performed automatically after each model

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) for the excellent face recognition library
- [MediaPipe](https://mediapipe.dev/) for efficient face detection
- [InsightFace](https://github.com/deepinsight/insightface) for ArcFace implementation
- [LFW Dataset](http://vis-www.cs.umass.edu/lfw/) for evaluation benchmarks
