# Aegis-FL: Autonomous Edge Intrusion Simulation in Federated Learning

[![Streamlit App]](https://aegis-fl.streamlit.app/)

Aegis-FL is a research-grade framework demonstrating Byzantine Fault Tolerance (BFT) defenses against model poisoning attacks in edge-based security networks.





## Overview
Federated Learning (FL) allows edge devices to collaboratively train an anomaly detection autoencoder without sharing raw data. However, FL is highly vulnerable to **Data and Model Poisoning** (Byzantine attacks). 

Aegis-FL simulates a network of benign and malicious clients and evaluates different aggregation strategies (FedAvg, FedMedian, FedKrum) to defend the global pipeline from network collapse.

## Features
- **PyTorch Edge Autoencoder**: Detects network anomalies from structured IoT traffic telemetry.
- **GenAI Sentinel Integration**: Edge nodes run highly-compressed mocked LLMs triggered by autoencoder breaches. Simulates real-world hardware latency (`time.sleep(1.5)`) dynamically on malicious traffic detection.
- **Threat Injector**: Deterministically simulates malicious nodes injecting sign-flipped weights to sabotage the global model (`src.adversary.attacks.SignFlipAttack`).
- **Byzantine Fault Tolerance**: Implements `FedKrum` and `FedMedian` to filter out malicious gradients.
- **Global Model Persistence**: Automatically intercepts and securely saves the final aggregated weights (`global_model_<strategy>.npz`) at the completion of round 10 for enterprise deployment.
- **Real-Time Telemetry**: A Streamlit dashboard to visualize global model loss, anomaly detection accuracy, and track attack impact across FL rounds independently per simulation run.

## 📦 Installation
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

## 🏃 Simulation Usage
Start a simulation with 5 clients, 2 of which are malicious, using the vulnerable `FedAvg` strategy:
```bash
python experiments/simulate.py --clients 5 --malicious 2 --strategy FedAvg
```

Run the defense simulation with `FedKrum`:
```bash
python experiments/simulate.py --clients 5 --malicious 2 --strategy FedKrum
```

Launch the Telemetry Dashboard (in a new terminal):
```bash
streamlit run dashboard/app.py
```

## 📄 Documentation
Check the `docs/` folder for the technical whitepaper detailing the experiment methodology and results.
