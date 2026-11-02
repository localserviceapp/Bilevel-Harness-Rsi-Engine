# Bi-Level Harness RSI Engine (v3)

A production-ready framework optimizing Recursive Self-Improvement (RSI) loops for Large Language Models. This architecture implements a specialized **Bi-Level Harness Optimization Engine** that safely manages self-directed evaluation, code execution, and programmatic self-updates over sequential validation cycles.

---

## 🚀 Core Innovations & AI Mathematical Framework

Unlike standard iterative generation pipelines, this engine introduces advanced mathematical structures to prevent alignment drift and evaluation degradation during recursive self-improvement.

### 🛡️ The KL Divergence Estimator Framework
To maintain historical optimization bounds across self-modification intervals, the engine leverages specialized KL estimators to compute distribution shifts during alignment phases:
*   **k1 / k2 / k3 Structural Regularization:** Dynamically enforces structural penalties during optimization steps to guarantee policy stability.
*   **Bi-Level Harness Convergence Proofs:** Provides mathematical safeguards against catastrophic collapse or policy degradation across multi-generation self-improvement operations.

---

## 🛠️ Architecture & Core Components

The repository is organized cleanly at the root level to support containerized execution and standard validation tracking:

*   📂 `rsi_engine/` — Core orchestration engine executing the 4-loop recursive self-improvement iterations.
*   📂 `tests/` — Automated test suites establishing evaluation constraints before code integration.
*   📂 `curriculum/` — Structured optimization metrics and dataset paths governing evaluation phases.
*   📄 `main.py` — Primary runtime entry point initializing the recursive loops.
*   📄 `Dockerfile` — Complete multi-stage build blueprint for isolated, stateless code execution.

---

## ⚙️ Quick Start & Environment Configuration

### Prerequisites
*   Python 3.10+ or Docker
*   A valid Google AI Studio Personal Access Token (or native environment endpoint keys)

### 1. Environment Set Up
Clone the repository and set your target execution keys:

```bash
# Clone the repository
git clone https://github.com
cd Bilevel-Harness-Rsi-Engine

# Export environment variables for the Gemini API model configurations
export GEMINI_API_KEY="your_personal_access_token_here"
```

### 2. Local Execution
Run the system natively using the primary entry point:
```bash
python main.py
```

### 3. Containerized Runtime (Recommended)
To isolate programmatic self-modification routines safely within an isolated environment, run via Docker:
```bash
# Build the multi-stage environment
docker build -t rsi-engine-v3 .

# Run the runtime matrix safely isolated
docker run -e GEMINI_API_KEY=$GEMINI_API_KEY rsi-engine-v3
```

---

## 🧪 Testing and Validation

Validate optimization thresholds and component logic using the integrated test suite:
```bash
pytest tests/
```

---

## 📖 Citation & Academic Reference

If you build upon this framework or utilize the bi-level harness optimization proofs in recursive self-improvement architectures, please cite the core engine work:

```text
Gavares, A. J. (2026). Mathematical Estimators and Bi-Level Harness Optimization Frameworks for Recursive Self-Improvement Engines. v3 Release Configuration.
```

