# Qbm-inspired-AI-in-Marin-port-Security-for-Spoofing
AI-Driven Maritime Blockchain Framework
Overview

This repository presents an AI-driven maritime blockchain framework designed to analyze, validate, and secure AIS-based maritime traffic data.
The primary goal of this research is to detect anomalous vessel behaviors from raw AIS time-series data and to record only AI-validated state transitions in a blockchain-inspired verification layer.

Unlike conventional approaches that treat blockchain as a passive data storage mechanism, this work explicitly models blockchain as a consensus layer driven by AI-derived trust and anomaly evidence.
The system is intended for research in maritime security, anomaly detection, data provenance, and trustworthy cyber-physical systems.

# Research Motivation

Maritime AIS data is vulnerable to noise, spoofing, missing signals, and intentional manipulation.
Simply storing raw AIS messages on-chain does not guarantee trustworthiness.

This research addresses the problem by introducing:

Behavior-aware AI models that interpret vessel dynamics over time

Explicitly engineered maritime anomaly features, rather than generic sequence embeddings

A verification-first blockchain architecture, where only validated behavioral transitions are committed

The resulting framework enables tamper-resistant, auditable, and explainable maritime event tracking.


# How to Run
1. Environment Setup

Python 3.11 or later is recommended.

conda create -n maritime-ai python=3.9
conda activate maritime-ai
pip install -r requirements.txt

2. Data Preparation

Place raw AIS data in the following directory:

data/raw/


Run the preprocessing and feature extraction pipeline:

python prepare_data.py


This step converts raw AIS messages into windowed time-series feature datasets stored in:

data/processed/


Note: This process may take several hours depending on dataset size.

3. Model Training

Train the anomaly detection model:

python train.py


Model checkpoints and logs will be saved automatically.

4. Evaluation and Analysis

Run evaluation scripts to generate metrics and visual outputs:

python experiments/run_pipeline.py --config configs/experiments/s0.yaml --ports ports/ports.yaml
python experiments/run_pipeline.py --config configs/experiments/s1.yaml --ports ports/ports.yaml
python experiments/run_pipeline.py --config configs/experiments/s2.yaml --ports ports/ports.yaml
python experiments/run_pipeline.py --config configs/experiments/s3.yaml --ports ports/ports.yaml
python experiments/calibrate_policy.py
python experiments/evaluate_end2end.py



Results will be stored in:

results/


# Research Status

This repository is an active research codebase.
Interfaces, features, and model configurations may change as experiments evolve.

The code is not intended for production deployment without further validation.

# License & Copyright

© UCS LAB. All rights reserved.

This code and associated research materials are the intellectual property of UCS LAB.
Unauthorized copying, redistribution, or commercial use without explicit permission is prohibited.

For academic or collaborative inquiries, please contact the authors through UCS LAB.

# Citation

If you use this code in academic work, please cite the corresponding paper (to be updated).
