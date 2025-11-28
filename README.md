# SEIZURE DETECTION USING QUANTIZED, CHANNEL-PRUNED & SNNS

## Setup Instructions

### Environment Setup - using uv

This project uses `uv` for fast and reliable dependency management.

1.  Install `uv` from the [installation](https://docs.astral.sh/uv/getting-started/installation/) page.
2.  Run `uv sync` in the root directory of the project. A `.venv` virtual environment folder will be created.
    ```bash
    uv sync
    ```
3.  Activate the virtual environment in your terminal:
    ```bash
    source .venv/bin/activate
    ```
4.  Alternatively, select the newly created `.venv` interpreter within your IDE (e.g., VS Code, PyCharm).

## Project Files and Execution

The core implementation and various analyses are distributed across the following files:

### 1. Main Execution

- **`final.py`**: This is the primary Python script that runs the full experimental pipeline, including training the baseline CNN, converting to SNN, applying channel/weight pruning, and running the various INT8 quantization strategies.
    ```bash
    python final.py
    ```

### 2. Notebooks for Analysis and Development

- **`final.ipynb`**: This Jupyter Notebook is used for **visualization of the final runs and results**. It generates the tables and figures (like Table 1, Table 2) mentioned in the paper to compare model efficiency (latency, energy, size) and performance (AUC, accuracy) across all configurations.

- **`ChannelAnalysis.ipynb`**: **channel analysis** steps described in Section 2.1, including calculating channel coverage and selecting the final 18-channel input feature set.


