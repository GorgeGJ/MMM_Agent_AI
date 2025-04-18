# MMM Agent (Streamlit + Ollama)

A powerful, local-first AI assistant for exploring Marketing Mix Modeling (MMM) using Bayesian modeling with PyMC and LLM-driven interaction via Ollama.

---

## 🔧 Features

✅ Upload your own media spend + revenue dataset (CSV)  
✅ Automatically retrains a PyMC-based Bayesian model on upload  
✅ Natural language agent powered by Ollama (e.g., llama3)  
✅ ROI summary with download option  
✅ Model diagnostics (trace plots, R-hat)  
✅ Interactive Streamlit UI

---

## 🗂️ CSV Format

Your CSV should contain columns named exactly like:
```
facebook, paid_search, youtube, revenue
```

---

## 🚀 How to Run Locally

### 1. Clone the repo and install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Ollama and load a model
```bash
ollama run llama3
```

### 3. (Optional) Generate a sample trace from built-in sample data
```bash
python generate_pymc_trace.py
```

### 4. Launch the app
```bash
python3 -m streamlit run mmm_agent_app.py
```

---

## 💬 Example Prompts to Try

- What was the ROI of Facebook in March 2024?
- Forecast sales if I spend $200K on Paid Search
- How should I allocate budget to maximize sales?

---

## 📤 Outputs

- `mmm_model_summary.csv`: ROI summary per channel
- Downloadable CSV from Streamlit interface
- Trace plots and R-hat diagnostics viewable in-app

---

## 🧪 Simulated Dataset

Want to test large-scale modeling? Use the [simulated 10K dataset](simulated_mmm_input_10000.csv) or generate your own via:
```bash
python simulate_data.py
```
