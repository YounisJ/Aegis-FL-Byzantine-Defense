import streamlit as st
import pandas as pd
import json
import time
import os
import plotly.express as px

st.set_page_config(page_title="Aegis-FL Dashboard", page_icon="🛡️", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🛡️ Aegis-FL")
    st.markdown("### Simulation Controls")
    st.markdown("Use the CLI to launch simulations:")
    st.code(
        "python experiments/simulate.py --clients 5 --malicious 2 --strategy FedAvg"
    )
    st.code(
        "python experiments/simulate.py --clients 5 --malicious 2 --strategy FedKrum"
    )

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ About Aegis-FL")
    st.markdown(
        "Aegis-FL is a research framework demonstrating **Byzantine Fault Tolerance** "
        "defenses against model poisoning in edge-based Federated Learning networks.\n\n"
        "It uses a mock **GenAI Sentinel** to simulate the inference hardware latency "
        "when malicious traffic anomalies are detected by the autoencoder."
    )
    st.markdown("Created by **[YounisJ](https://github.com/YounisJ)**")

st.title("🛡️ Aegis-FL: Byzantine Fault Tolerance in GenAI Sentinels")
st.markdown(
    "Real-time telemetry for Federated Learning model training under adversarial conditions."
)


# Helper to read JSONL
def load_data(log_file=None):
    if log_file is None:
        log_file = os.path.join(os.path.dirname(__file__), "metrics.jsonl")
    if not os.path.exists(log_file):
        return pd.DataFrame(
            columns=[
                "timestamp",
                "run_id",
                "round",
                "strategy",
                "loss",
                "accuracy",
                "attack_success",
            ]
        )

    data = []
    with open(log_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
    df = pd.DataFrame(data)
    if not df.empty and "strategy" not in df.columns:
        df["strategy"] = "Unknown"
    return df


st.empty()  # For structure
df = load_data()

tab1, tab2, tab3 = st.tabs(["📈 Live Telemetry", "🕵️‍♂️ Under The Hood", "📊 Raw Logs"])

with tab1:
    if df.empty:
        st.info(
            "No metrics available yet. Run a simulation using the CLI commands in the sidebar."
        )
    else:
        # Get the latest round for each strategy
        latest_runs_df = df.groupby("strategy").last().reset_index()

        # Filter the plotting dataframe to only the latest run_id per strategy
        latest_run_ids = latest_runs_df["run_id"].unique()
        plot_df = df[df["run_id"].isin(latest_run_ids)]

        for _, row in latest_runs_df.iterrows():
            strat = row["strategy"]
            latest_loss = row["loss"]
            latest_acc = row.get("accuracy", 1.0)

            # Status Banner
            if latest_acc < 0.80:
                st.error(
                    f"⚠️ **Model Collapse Detected ({strat})!** The attack successfully breached the threshold. Latest Acc: {latest_acc:.4f}, Loss: {latest_loss:.4f}"
                )
            else:
                st.success(
                    f"✅ **Model Stable ({strat}).** The defense successfully filtered malicious gradients. Latest Acc: {latest_acc:.4f}, Loss: {latest_loss:.4f}"
                )

            st.markdown("### Key Performance Indicators")
            # Metrics
            strat_df = plot_df[plot_df["strategy"] == strat]
            if len(strat_df) >= 2:
                prev_acc = strat_df.iloc[-2].get("accuracy", 1.0)
                prev_loss = strat_df.iloc[-2].get("loss", 0.0)
                acc_delta = latest_acc - prev_acc
                loss_delta = latest_loss - prev_loss
            else:
                acc_delta = None
                loss_delta = None

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Round", f"{int(row['round'])} / 10")
            c2.metric(
                "Global Accuracy",
                f"{latest_acc:.2%}",
                delta=f"{acc_delta:.2%}" if acc_delta is not None else None,
            )
            c3.metric(
                "Global Loss",
                f"{latest_loss:.4f}",
                delta=f"{loss_delta:.4f}" if loss_delta is not None else None,
                delta_color="inverse",
            )
            c4.metric("Attack Success Rate", f"{row.get('attack_success', 0.0):.2%}")

        st.markdown("---")

        st.subheader("Global Anomaly Detection Accuracy")
        st.markdown(
            "If the accuracy drops below the **red dashed line**, the global model has been successfully poisoned."
        )

        if "accuracy" in plot_df.columns:
            fig_acc = px.line(
                plot_df,
                x="round",
                y="accuracy",
                color="strategy",
                markers=True,
                template="plotly_dark",
            )
            fig_acc.add_hline(
                y=0.80,
                line_dash="dash",
                line_color="red",
                annotation_text="Collapse Threshold",
            )
            fig_acc.update_layout(yaxis_title="Accuracy", xaxis_title="FL Round")
            st.plotly_chart(fig_acc, use_container_width=True, key="acc_chart")

        st.subheader("Global Model Loss over FL Rounds")
        fig_loss = px.line(
            plot_df,
            x="round",
            y="loss",
            color="strategy",
            markers=True,
            template="plotly_dark",
        )
        fig_loss.update_layout(yaxis_title="Loss (MSE)", xaxis_title="FL Round")
        st.plotly_chart(fig_loss, use_container_width=True, key="loss_chart")

with tab2:
    st.markdown(
        """
    ## 🕵️‍♂️ How Aegis-FL Works
    
    ### 1. Federated Learning (FL)
    Instead of sending raw network traffic data to a centralized server, edge nodes collaboratively train a local **PyTorch Autoencoder** to detect anomalies. Only the model weights (gradients) are sent to the central server.
    
    ### 2. The Byzantine Threat 😈
    A malicious node can attempt to poison the global model by sending mathematically sabotaged weights (`SignFlipAttack`). If the server naively averages the weights (`FedAvg`), the entire global model collapses and becomes useless.
    
    ### 3. The Defense 🛡️
    By using advanced Byzantine-robust aggregation strategies like **FedKrum** or **FedMedian**, the central server can mathematically identify and discard the poisoned weights, keeping the global model stable.
    
    ### 4. GenAI Sentinels 🤖
    To make this simulation realistic, we assume edge nodes are running highly compressed Large Language Models (LLMs) to perform deep semantic analysis when the Autoencoder flags suspicious packets. Because running an LLM on an edge device is computationally expensive, we accurately simulate this hardware latency dynamically during the evaluation phase whenever malicious activity is detected!
    """
    )

with tab3:
    st.markdown("### Raw Telemetry Logs")
    st.dataframe(df, use_container_width=True)

# Auto-refresh polling (only runs locally, but gives live feedback)
time.sleep(2)
st.rerun()
