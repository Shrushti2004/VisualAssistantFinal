# Video Understanding & Question-Answering System

## Overview
This repository contains a video understanding and question-answering system capable of:
- Detecting objects in video frames
- Recognizing actions using VideoMAE
- Answering both descriptive and multiple-choice questions based on video content
- Handling large videos efficiently with GPU-optimized batch processing

The system is built with FastAPI for serving the API and Groq LLM for reasoning.

---

## Features
- **Object Detection**: YOLOv8x-based detection across sampled frames
- **Action Recognition**: VideoMAE-based classification
- **Scene Summary Generation**: Combines object and action information
- **QA Support**: Answers descriptive and MCQs
- **Batch Processing**: Optimized GPU usage
- **Memory-Safe**: Works for videos of any length

---

## Architecture Diagram

**Flow:**
1. Video input → Frame extraction (Decord/OpenCV)
2. Object detection on sampled frames (YOLOv8x)
3. Action recognition (VideoMAE)
4. Scene summary generation
5. Question answering via Groq API (MCQ or descriptive)
6. Return answer in plain text via FastAPI

---

## Tech Stack
| Component | Technology | Justification |
|-----------|------------|---------------|
| Object Detection | YOLOv8x | High accuracy real-time detection |
| Action Recognition | VideoMAE | Efficient and reliable video action classification |
| Frame Extraction | OpenCV, Decord | Handles videos of any length |
| API Framework | FastAPI | Lightweight, high-performance REST API |
| QA Backend | Groq API | LLM reasoning for MCQ and descriptive answers |
| GPU Optimization | CUDA + batch processing | Maximizes GPU throughput and memory efficiency |

---

## Installation
```bash
git clone <your-repo-url>
cd video_infer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


🎥 Demo Video: https://drive.google.com/drive/folders/1bW5Z6Y3t7V8oPgGQ43uTR-31nXd-UB-4

🖼️ Architecture Diagram: https://drive.google.com/file/d/1rmGZmFwCq3SMG1ZZC1M-DhNfkjcqdtin/view?usp=sharing




