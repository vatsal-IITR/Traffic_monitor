AI Traffic Monitoring System

Overview

This project implements an intelligent traffic monitoring system using Computer Vision. It detects vehicles, tracks them, counts entries, estimates congestion, and generates adaptive signal timing.

Features

YOLO-based vehicle detection
ByteTrack tracking (no duplicate counting)
Lane-wise vehicle entry counting
Real-time traffic density estimation
Adaptive traffic signal timing
Heatmap generation
Interactive dashboard (Streamlit)
Methodology

Input traffic video
Detect vehicles using YOLOv8
Track vehicles using ByteTrack
Count vehicles using line-crossing
Estimate density using lane regions
Generate heatmaps
Display results in dashboard
How to Run

Step 1: Install dependencies

pip install -r requirements.txt

Step 2: Run main system

python main.py

Step 3: Launch dashboard

streamlit run dashboard.py

Output

CSV file with traffic statistics
Heatmaps
Interactive dashboard with graphs and metrics
Evaluation Metrics

Precision
Recall
F1 Score
Accuracy
Future Work

Real-time deployment
Emergency vehicle prioritization
Smart city integration
