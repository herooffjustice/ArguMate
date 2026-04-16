# ArguMate — Contrarian Teaching Assistant

> *"Well, not exactly..."* — ArguMate never lets you get away with a half-baked answer.

ArguMate is an AI-powered teaching assistant that **argues back**. You submit an RNN/deep learning concept and your understanding of it — ArguMate uses an NLP pipeline to evaluate your answer, then has LLaMA 3.3 70B (via Groq) push back, correct misconceptions, or force you to think deeper. It's Socratic learning, but disagreeable.

***

## How It Works

1. **You ask** about an RNN concept (hidden states, BPTT, LSTMs, attention, etc.)
2. **You explain** what you think you understand
3. **ArguMate's NLP pipeline** evaluates your explanation using:
   - **TF-IDF** for fast keyword-based intent detection
   - **Sentence-BERT** (`all-MiniLM-L6-v2`) for semantic similarity scoring
   - **VADER** sentiment analysis to detect your confidence level
4. **LLaMA 3.3 70B** (via Groq) generates a contrarian, Socratic response — pushing back harder if you're wrong, and harder still if you're right (no free passes)
5. **Full conversation memory** is maintained per session

### NLP Pipeline

```
User input ──► Intent Detection (TF-IDF + Sentence-BERT)
                        │
                        ▼
              Semantic Similarity Score
              vs. Ground Truth Explanation
                        │
                  [CORRECT / INCORRECT]
                        │
              Sentiment Analysis (VADER)
              [confident / neutral / uncertain]
                        │
                        ▼
              Groq API → LLaMA 3.3 70B
              (system prompt + session history)
                        │
                        ▼
              Contrarian pedagogical response
```

### Topics Covered

ArguMate has a built-in knowledge base covering **35+ RNN concepts**, including:

| Category | Topics |
|---|---|
| RNN Fundamentals | Architecture, hidden states, unrolling, BPTT, parameter sharing |
| Gradient Problems | Vanishing gradients, exploding gradients, gradient clipping |
| Gated Architectures | LSTM (all gates + equations), GRU, GRU vs LSTM comparison |
| Sequence Modeling | Seq2Seq, context vectors, teacher forcing, encoder-decoder |
| Attention | Bahdanau attention, Luong attention, alignment scores |
| Advanced Topics | BiRNNs, deep/stacked RNNs, RNN language models, perplexity, dropout |

***

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| NLP Pipeline | `sentence-transformers` (MiniLM), `scikit-learn` (TF-IDF), `vaderSentiment` |
| LLM | LLaMA 3.3 70B Versatile via [Groq API](https://groq.com) |
| Frontend | HTML/CSS/JS (served via Flask templates) |
| Deployment | Render (free tier) |

***

## Setup & Run

### Prerequisites

- Python 3.9+
- A free [Groq API key](https://console.groq.com)

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/herooffjustice/ArguMate.git
cd ArguMate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
export GROQ_API_KEY=your_key_here          # Mac/Linux
set GROQ_API_KEY=your_key_here             # Windows CMD
$env:GROQ_API_KEY="your_key_here"          # Windows PowerShell

# 4. Run
python app.py
```

Then open [http://localhost:5000](http://localhost:5000)

> **Note:** The first request triggers lazy model loading (Sentence-BERT downloads ~80MB). Subsequent requests are fast.

***

## Deploy to Render (Free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo
3. Set the following:

| Field | Value |
|---|---|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `python app.py` |
| **Environment Variable** | `GROQ_API_KEY` = your key |

***

## API Reference

### `POST /chat`

```json
{
  "session_id": "user-123",
  "question": "What is the vanishing gradient problem?",
  "understanding": "I think it happens when gradients get too small during backprop"
}
```

**Response:**
```json
{
  "response": "Well, 'too small' is doing a lot of work there...",
  "label": "CORRECT",
  "similarity": 0.7142,
  "tone": "uncertain",
  "concept": "vanishing_gradient"
}
```

### `POST /reset`

Clears the conversation history for a session.

```json
{ "session_id": "user-123" }
```

### `GET /health`

Returns model load status.

***

## Project Structure

```
ArguMate/
├── app.py              # Flask app, NLP pipeline, Groq integration
├── templates/
│   └── index.html      # Chat UI
├── requirements.txt
├── .env.example        # Copy to .env and fill in your key
└── .gitignore
```

***

## License

MIT — use it, fork it, argue with it.
