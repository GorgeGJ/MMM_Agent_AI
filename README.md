# 🧠 MMM Agent AI

A natural language interface for your Marketing Mix Modeling (MMM) system, powered by LangChain and Streamlit. This agent lets stakeholders run historical performance queries and scenario simulations using plain English.

## 💡 Features

- Ask questions like:
  - _"How much did YouTube contribute in 2024?"_
  - _"If I invest $500K in Paid Social next quarter, what would happen?"_

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/mmm-agent-ai.git
cd mmm-agent-ai
```

### 2. Setup

```bash
cp .env.example .env
```

Add your OpenAI API key to `.env`.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

## 📂 Structure

- `app.py`: Main Streamlit app
- `agent/tools.py`: MMM tools
- `.env.example`: Env template
- `requirements.txt`: Dependencies
