import torch
import torch.nn as nn
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

import warnings

# Suppress HuggingFace warnings for cleaner logs
warnings.filterwarnings("ignore", module="transformers")


class EdgeAutoencoder(nn.Module):
    """
    A lightweight Autoencoder designed for IoT edge anomaly detection.
    Compresses network traffic/telemetry down to a latent space and attempts reconstruction.
    High reconstruction error indicates an anomaly (attack).
    """

    def __init__(self, input_dim=41, latent_dim=16):
        super(EdgeAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),  # Features are scaled 0-1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class GenAISentinel:
    """
    Combines the Edge Autoencoder with a lightweight LLM.
    If the Autoencoder detects an anomaly, the LLM is queried to classify/explain the threat.
    """

    def __init__(
        self, input_dim=41, llm_model_name="distilgpt2", load_in_4bit=False
    ):
        self.autoencoder = EdgeAutoencoder(input_dim=input_dim)
        self.threshold = 0.05  # MSE threshold for anomaly
        self.llm_model_name = llm_model_name
        self.load_in_4bit = load_in_4bit

        # We don't load the LLM into memory during FL training to save RAM,
        # but we mock the interface for the architecture.
        self.llm_pipeline = None

    def initialize_llm(self):
        self.llm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.pipeline = None
        logging.info(f"Initializing LLM Sentinel ({self.llm_model_name})...")
        try:
            from transformers import pipeline
        except ImportError:
            logging.warning(
                "transformers not installed. LLM functionality disabled."
            )
            return

        if self.load_in_4bit:
            logging.info("Loading in 4-bit precision for extreme edge.")
            # Note: requires bitsandbytes and accelerate. Mocked for this demo.
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name
            )
            self.llm_pipeline = pipeline(
                "text-generation", model=self.model, tokenizer=self.tokenizer
            )

    def process_packet(self, x_tensor, dynamic_threshold=None):
        """
        Process a network packet (tensor).
        Returns a tuple: (is_anomaly, llm_explanation)
        """
        self.autoencoder.eval()
        threshold = (
            dynamic_threshold
            if dynamic_threshold is not None
            else self.threshold
        )
        with torch.no_grad():
            reconstructed = self.autoencoder(x_tensor)
            mse = torch.mean((x_tensor - reconstructed) ** 2).item()

            is_anomaly = mse > threshold

        # Simulating Edge LLM latency overhead
        if is_anomaly:
            time.sleep(1.5)

            if self.llm_pipeline is None:
                return (
                    True,
                    f"Mocked LLM Alert: High MSE {mse:.4f}. Suspected intrusion.",
                )
            else:
                prompt = f"Network anomaly detected with MSE {mse:.4f}. What is the likely attack vector?"
                response = self.llm_pipeline(
                    prompt, max_length=50, num_return_sequences=1
                )
                return True, response[0]["generated_text"]

        return False, "Normal traffic."


if __name__ == "__main__":
    sentinel = GenAISentinel()
    mock_packet = torch.rand(1, 41)
    is_anomaly, explanation = sentinel.process_packet(mock_packet)
    logging.info(f"Anomaly: {is_anomaly}, Explanation: {explanation}")
