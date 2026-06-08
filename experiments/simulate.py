import subprocess
import time
import sys
import argparse
import uuid


def start_simulation(
    num_clients=5,
    num_malicious=1,
    strategy="FedAvg",
    seed=None,
    clear=True,
    port=8080,
):
    print("--- Starting Aegis-FL Simulation ---")
    print(f"Strategy: {strategy}")
    print(f"Total Clients: {num_clients}")
    print(f"Malicious Clients: {num_malicious}")

    processes = []

    # 1. Start Server
    run_id = str(uuid.uuid4())
    seed_arg = f"{seed}" if seed is not None else "None"
    server_cmd = [
        sys.executable,
        "-c",
        f"from src.fl_server.server import start_server; start_server('{strategy}', {num_malicious}, clear={clear}, run_id='{run_id}', seed={seed_arg}, port={port})",
    ]
    print(f"Launching Server (Run ID: {run_id})...")
    server_proc = subprocess.Popen(server_cmd)
    processes.append(server_proc)

    # Wait for server to initialize
    time.sleep(3)

    # 2. Start Clients
    for i in range(num_clients):
        is_malicious = i < num_malicious
        client_cmd = [
            sys.executable,
            "-m",
            "src.fl_client.client",
            "--id",
            str(i),
            "--port",
            str(port),
        ]
        if is_malicious:
            client_cmd.append("--malicious")
        if seed is not None:
            client_cmd.extend(["--seed", str(seed)])

        print(f"Launching Client {i} (Malicious: {is_malicious})...")
        client_proc = subprocess.Popen(client_cmd)
        processes.append(client_proc)
        time.sleep(0.5)

    try:
        # Keep alive while server and clients run
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("Simulation stopped by user. Cleaning up...")
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clients", type=int, default=5, help="Total number of clients"
    )
    parser.add_argument(
        "--malicious", type=int, default=1, help="Number of malicious clients"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="FedAvg",
        choices=["FedAvg", "FedKrum", "FedMedian"],
        help="FL Strategy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        default=True,
        help="Clear metrics log before starting",
    )
    parser.add_argument(
        "--no-clear",
        dest="clear",
        action="store_false",
        help="Do not clear metrics log",
    )
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    args = parser.parse_args()

    start_simulation(
        args.clients,
        args.malicious,
        args.strategy,
        args.seed,
        args.clear,
        args.port,
    )
