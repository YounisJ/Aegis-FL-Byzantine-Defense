# Aegis-FL: Securing Federated Sentinels Against Byzantine Poisoning
**Research Draft**

## Abstract
As Machine Learning models are increasingly deployed at the edge for real-time cyber threat intelligence, the need to fine-tune them collaboratively using Federated Learning (FL) has grown. However, FL exposes the system to Byzantine faults, where compromised nodes inject malicious weight updates to collapse the global model. In this paper, we present **Aegis-FL**, a simulation framework demonstrating the vulnerability of edge sentinels to model poisoning and evaluating the efficacy of distance-based aggregation defenses like `FedKrum` and `FedMedian`.

## 1. Introduction
Edge-based Autoencoders provide highly localized, privacy-preserving anomaly detection. Fine-tuning these sentinels collaboratively via FL is highly efficient but fundamentally assumes all clients are trustworthy. We simulate a scenario where $K$ out of $N$ clients are compromised by an adversary attempting a targeted sign-flipping attack to maximize the global model's Mean Squared Error (MSE), effectively blinding the anomaly detection system.

## 2. Threat Model & Methodology
- **Network**: $N = 5$ IoT edge nodes.
- **Model Architecture**: PyTorch Autoencoder (Input 41 -> Latent 16) trained on structured synthetic IoT telemetry.
- **Adversary**: Deterministic Threat Injector altering the gradient updates of $K$ nodes ($\theta_{malicious} = \theta_{benign} \times -5.0$).
- **Defenses Evaluated**: `FedAvg` (Baseline) vs. `FedKrum` (Byzantine-Resilient).

## 3. Preliminary Results
When running under `FedAvg` with $K=2$, the global model MSE diverges immediately, resulting in an elevated plateau oscillating at ~0.30, dropping anomaly detection accuracy significantly. By substituting the aggregation engine with `Krum` (FedKrum), the server successfully filters out the statistically distant malicious updates. The global model loss converges smoothly, maintaining high anomaly detection accuracy, matching the baseline $K=0$ scenario, despite the presence of 40% malicious actors.

## 4. Conclusion & Future Work
We demonstrate that naive FL aggregation is insufficient for critical edge security. While `FedKrum` successfully defends against sign-flipping, future work will benchmark the latency overhead of distance calculations on full-scale parameter matrices at the edge orchestrator and introduce more sophisticated attacks like targeted label-flipping.
