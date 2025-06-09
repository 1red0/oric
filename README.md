# ORIC - Object Recognition & Image Classification

[![Deployed on Vercel](https://img.shields.io/badge/Deployed-Vercel-black?logo=vercel)](https://oric-ml.vercel.app/)
[![Next.js](https://img.shields.io/badge/Next.js-15+-black?logo=next.js)](https://nextjs.org/)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.0+-orange?logo=tensorflow)](https://www.tensorflow.org/js)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue?logo=typescript)](https://www.typescriptlang.org/)

## 🎯 Overview

ORIC is a modern web application that demonstrates the power of computer vision using machine learning models directly in the browser. Built for a Master's thesis project, it showcases real-time object detection and image classification capabilities using TensorFlow.js and Hugging Face models.

**🚀 [Live Demo](https://oric-ml.vercel.app/)**

## ✨ Features

### 🎯 Object Recognition

Detect and locate multiple objects within images with precise bounding boxes and confidence scores.

- **Real-time object detection** with visual bounding box overlays
- **Multi-object detection** - identify multiple objects in a single image
- **Confidence scoring** - quantitative assessment of detection accuracy
- **Spatial localization** - precise pixel-level object positioning
- **Category classification** - objects labeled with specific class names

**Available Object Detection Models:**

| Model | Architecture | Objects Detected | Strengths | Use Case |
|-------|-------------|------------------|-----------|----------|
| **COCO-SSD** | Single Shot Detector | 80 COCO classes | Fast inference, mobile-friendly | Real-time applications |
| **DETR ResNet-50** | Detection Transformer | 80+ object classes | Balanced accuracy/speed | General-purpose detection |
| **DETR ResNet-101** | Detection Transformer | 80+ object classes | High accuracy, detailed detection | Precision-critical tasks |
| **YOLOS Tiny** | Vision Transformer | 80+ object classes | Ultra-fast, lightweight | Edge computing, mobile |

### 🖼️ Image Classification

Identify objects, scenes, and concepts in images with detailed confidence scores and category predictions.

- **Comprehensive scene understanding** - classify overall image content
- **Hierarchical classification** - from general to specific categories
- **Semantic analysis** - understand image context and meaning
- **Probability distribution** - complete confidence breakdown across classes

**Available Image Classification Models:**

| Model | Architecture | Classes | Parameters | Strengths | Best For |
|-------|-------------|---------|------------|-----------|----------|
| **MobileNet** | Depthwise Separable CNN | 1,000 ImageNet | 4.2M | Extremely fast, mobile-optimized | Mobile apps, real-time |
| **ResNet-50** | Residual Network | 1,000 ImageNet | 25.6M | Balanced performance | General classification |
| **ResNet-101** | Residual Network | 1,000 ImageNet | 44.5M | High accuracy, robust | Detailed analysis |
| **ResNet-152** | Residual Network | 1,000 ImageNet | 60.2M | Maximum accuracy | Research, benchmarking |

## 🛠️ Technology Stack

- **Frontend Framework**: Next.js 13+ with App Router
- **Language**: TypeScript
- **Machine Learning**: TensorFlow.js
- **Model Sources**:
  - Hugging Face Transformers (Object Detection)
  - TensorFlow Hub & Pre-trained Models (Image Classification)
- **Styling**: Tailwind CSS
- **Deployment**: Vercel
- **Build Tool**: Turbopack (Next.js)

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        ORIC Application                         │
├─────────────────────────────────────────────────────────────────┤
│                     User Interface Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Image Upload    │  │ Model Selection │  │ Results Display │  │
│  │ • Drag & Drop   │  │ • Object Detect │  │ • Bounding Box  │  │
│  │ • File Browser  │  │ • Image         |  | • Confidence    |  |
|  |                 |  |   Classification│  │ • Category List │  │
│  │                 │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Processing Engine                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Object Detection│  │Image Classificat│  │ Model Manager   │  │
│  │ • COCO-SSD      │  │ • MobileNet     │  │ • Model Loading │  │
│  │ • DETR R-50/101 │  │ • ResNet Series │  │                 │  │
│  │ • YOLOS Tiny    │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                   TensorFlow.js Runtime                         │
│  ┌─────────────────┐                                            │
│  │ WebGL Backend   │                                            │
│  │ • GPU           |                                            |
|  |   Acceleration  |                                            |
│  └─────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                   ┌──────────────────────┐
                   │    Model Sources     │
                   │ ┌──────────────────┐ │
                   │ │ Hugging Face Hub │ │
                   │ │ • DETR Models    │ │
                   │ │ • YOLOS Models   │ │
                   │ └──────────────────┘ │
                   │ ┌──────────────────┐ │
                   │ │ TensorFlow Hub   │ │
                   │ │ • ResNet Models  │ │
                   │ │ • MobileNet      │ │
                   │ └──────────────────┘ │
                   └──────────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- pnpm

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/1red0/oric.git
   cd oric
   ```

2. **Install dependencies**

   ```bash
   pnpm install
   ```

3. **Run the development server**

   ```bash
   pnpm dev
   ```

4. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

## 📖 Usage

### Object Recognition Workflow

1. **Select Detection Model**: Choose from COCO-SSD, DETR ResNet-50/101, or YOLOS Tiny
2. **Upload Target Image**: Drag and drop or browse for images (JPG, PNG, GIF, WebP)
3. **Process Detection**: Model analyzes image and identifies objects
4. **View Results**:
   - Bounding boxes drawn around detected objects
   - Object labels with confidence percentages

### Image Classification Workflow

1. **Select Classification Model**: Choose from MobileNet or ResNet variants (50/101/152)
2. **Upload Target Image**: Support for common image formats
3. **Process Classification**: Model analyzes entire image for content identification
4. **View Predictions**:
   - Top predictions with confidence scores
   - Detailed category breakdown

### Best Practices

#### For Object Recognition

- **Clear Images**: Use high-resolution images with well-defined objects
- **Good Lighting**: Ensure adequate contrast between objects and background
- **Multiple Objects**: Test with images containing 2-10 objects for optimal performance
- **Model Selection**:
  - Use COCO-SSD for speed-critical applications
  - Use DETR ResNet-101 for maximum accuracy
  - Use YOLOS Tiny for mobile/edge deployment

#### For Image Classification

- **Centered Subjects**: Place main subject in center of frame when possible
- **Clear Focus**: Avoid heavily blurred or abstract images
- **Standard Scenes**: Models perform best on common ImageNet categories
- **Model Selection**:
  - Use MobileNet for real-time mobile applications
  - Use ResNet-152 for research and detailed analysis
  - Use ResNet-50 for balanced performance

## 🏭 Building for Production

```bash
pnpm run build
pnpm start
```

NOTE: *The application is optimized for static deployment and can be deployed to Vercel*

## 🧪 Research Context

This project was developed as part of a Master's thesis exploring multiple dimensions of browser-based machine learning:

### Core Research Questions

1. **Client-Side ML Feasibility**: Can complex computer vision models run effectively in web browsers?
2. **Performance Trade-offs**: How do different model architectures compare in web environments?
3. **Comparison between features**: What are the differences between object detection and image classification?
4. **User Experience Design**: How can AI-powered interfaces be made intuitive and accessible?

### Research Contributions

#### Technical Contributions

- **Comprehensive Model Evaluation**: Systematic comparison of 8 different models across 2 computer vision tasks
- **Cross-Platform Compatibility**: Ensuring consistent performance across devices and browsers

#### Methodological Contributions

- **Benchmarking Framework**: Standardized evaluation metrics for web-based ML applications
- **Performance Profiling**: Inference speed, and accuracy measurement tools
- **Comparative Analysis**: Object detection vs. image classification effectiveness for different use cases

### Academic Impact

#### Research Areas Addressed

- **Edge Computing**: Bringing ML inference to client devices
- **Human-Computer Interaction**: AI interface design principles
- **Computer Vision**: Practical deployment of state-of-the-art models
- **Web Technology**: Advanced JavaScript ML frameworks and optimization

#### Practical Applications

- **Educational Tools**: Interactive ML demonstrations for computer vision courses
- **Rapid Prototyping**: Fast iteration on computer vision applications
- **Privacy-Preserving AI**: Local processing without data transmission
- **Accessibility**: Making advanced AI tools available through web browsers

## 📊 Performance Metrics & Benchmarks

### Model Performance Comparison

#### Object Detection Models

| Model | Avg. Inference (ms) | Best Use Case |
|-------|-------------------|---------------|
| **COCO-SSD** | 45-80 | Mobile, Real-time |
| **DETR ResNet-50** | 200-400 | Balanced Performance |
| **DETR ResNet-101** | 350-600 | High Accuracy |
| **YOLOS Tiny** | 80-120 | Edge Computing |

#### Image Classification Models

| Model | Avg. Inference (ms) | Top-1 Accuracy | Top-5 Accuracy |
|-------|-------------------|----------------|----------------|
| **MobileNet** | 15-25 | 70.4% | 89.5% |
| **ResNet-50** | 40-70 | 76.1% | 92.9% |
| **ResNet-101** | 70-120| 77.4% | 93.6% |
| **ResNet-152** | 100-180 | 78.3% | 94.1% |

NOTE: *Performance metrics measured on Chrome 120+ with hardware acceleration enabled*

### System Requirements & Optimization

#### Minimum Requirements

- **Browser**: Chrome 88+, Firefox 89+, Safari 14+, Edge 88+
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: WebGL 2.0 support recommended for optimal performance
- **JavaScript**: ES2020+ support required

### Benchmark Methodology

#### Testing Environment

- **Hardware**: Various devices from mobile to desktop
- **Browsers**: Cross-browser compatibility testing
- **Network**: Performance under different connection speeds
- **Images**: Standardized test dataset with varying complexity

## 📝 License

This project is part of academic research. Please refer to the license file for usage terms.

## 🔗 Links

- **Live Application**: <https://oric-ml.vercel.app/>
- **GitHub Repository**: <https://github.com/1red0/oric>
- **TensorFlow.js**: <https://www.tensorflow.org/js>
- **Hugging Face**: <https://huggingface.co/>

NOTE: *This project demonstrates the intersection of modern web technologies and machine learning, showcasing how sophisticated AI capabilities can be made accessible through intuitive web interfaces.*
