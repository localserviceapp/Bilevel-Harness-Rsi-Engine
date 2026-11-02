# Bi-Level Harness Optimization & Textual Backpropagation (4-Loop RSI Engine v3)

Reference implementation for **Bi-Level Harness Optimization and Textual Backpropagation in Autonomous 4-Loop Recursive Self-Improvement Engines** (Gavares & Gavares, 2026).

## Quickstart with Gemini (Free Tier / AI Studio)

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Export your key and run the 4-Loop Engine:
```bash
export GEMINI_API_KEY="your-api-key-here"
python3 main.py --demo --provider gemini \
  --loop1-model gemini-2.5-flash \
  --loop2-model gemini-2.5-pro \
  --export-jsonl verified_trajectories.jsonl
```

3. Or run in offline deterministic verification mode (no API key needed):
```bash
python3 main.py --demo --provider mock
python3 -m unittest discover -s tests -v
```

## Multi-Provider Support (`llm/llm_client.py`)
- **`gemini`**: Uses Google Gemini (`gemini-2.5-flash` for Loop 1 DAG execution, `gemini-2.5-pro` for Loop 2 Meta-Architect & Loop 3 Challenger). Zero external SDK required (uses native REST API via `GEMINI_API_KEY`).
- **`openai_compatible`**: Works with DeepSeek (`https://api.deepseek.com/v1`), DeepInfra (`Qwen/Qwen2.5-Coder-32B-Instruct`), OpenRouter, or OpenAI via `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
- **`ollama`**: Connects to local Ollama (`http://localhost:11434`) for $0 local open-weights execution.
- **`mock`**: Deterministic simulator for unit testing and CI verification.
