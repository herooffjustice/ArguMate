from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import numpy as np
import os
import threading

app = Flask(__name__)

# ── KNOWLEDGE BASE ─────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = {}

KNOWLEDGE_BASE["rnn_basics"] = {"aliases": ["recurrent neural network", "rnn", "what is rnn", "rnn architecture", "sequential model", "sequence model", "recurrent network"], "explanation": "A Recurrent Neural Network (RNN) is a neural network designed for sequential data. Unlike feedforward networks, RNNs have connections that loop back, allowing information to persist across timesteps via a hidden state. At each step t, the hidden state h_t = tanh(W_h * h_{t-1} + W_x * x_t + b), combining the current input and the previous hidden state."}
KNOWLEDGE_BASE["hidden_state"] = {"aliases": ["hidden state", "rnn memory", "h_t", "recurrent state", "internal memory rnn", "short term memory rnn", "state vector"], "explanation": "The hidden state h_t in an RNN is a fixed-size vector that summarizes information from all previous timesteps. It acts as the network's working memory, updated at each step by combining the previous hidden state and the current input. In LSTMs, it represents short-term memory while the cell state handles long-term memory."}
KNOWLEDGE_BASE["unrolling_rnn"] = {"aliases": ["unrolling rnn", "unfolding rnn", "rnn unrolled", "rnn through time", "rnn computational graph", "rnn timesteps"], "explanation": "Unrolling an RNN means expanding the recurrent computation graph across all timesteps, making it resemble a deep feedforward network where each layer corresponds to one timestep. This enables backpropagation through time (BPTT). The depth of the unrolled graph equals the sequence length, which is why long sequences cause gradient problems."}
KNOWLEDGE_BASE["parameter_sharing"] = {"aliases": ["parameter sharing rnn", "shared weights rnn", "same weights each step", "weight reuse rnn", "rnn weight sharing"], "explanation": "RNNs use the same weight matrices at every timestep — parameter sharing. It allows the model to generalize across different positions in a sequence and reduces parameters drastically. However, errors compound multiplicatively during backpropagation, contributing to vanishing and exploding gradients."}
KNOWLEDGE_BASE["rnn_types"] = {"aliases": ["types of rnn", "one to many rnn", "many to one rnn", "many to many rnn", "rnn architectures types", "rnn variants"], "explanation": "RNNs can be structured in several configurations: one-to-one (standard network), one-to-many (image captioning), many-to-one (sentiment analysis), many-to-many equal length (POS tagging), and many-to-many different lengths (machine translation via encoder-decoder). The choice depends on the task structure."}
KNOWLEDGE_BASE["bptt"] = {"aliases": ["backpropagation through time", "bptt", "rnn training algorithm", "how rnn is trained", "truncated bptt", "truncated backpropagation"], "explanation": "Backpropagation Through Time (BPTT) trains RNNs by unrolling the network across all timesteps and computing gradients using the chain rule. The gradient involves products of Jacobians across all timesteps. Truncated BPTT limits the number of timesteps to reduce memory and computation at the cost of not capturing very long-range dependencies."}
KNOWLEDGE_BASE["vanishing_gradient"] = {"aliases": ["vanishing gradient", "gradient vanishes", "gradient becomes zero", "long term dependency problem", "gradient shrinks", "rnn forgets long sequences"], "explanation": "The vanishing gradient problem occurs in RNNs during BPTT when gradients are multiplied by the recurrent weight matrix repeatedly. If singular values of the weight matrix are less than 1, gradients shrink exponentially for early timesteps, preventing the network from learning long-range dependencies. LSTMs and GRUs were designed to address this."}
KNOWLEDGE_BASE["exploding_gradient"] = {"aliases": ["exploding gradient", "gradient explodes", "nan loss rnn", "gradient clipping", "unstable rnn training", "gradient norm"], "explanation": "Exploding gradients occur when repeated multiplication of recurrent weights causes gradients to grow exponentially during BPTT, leading to large weight updates and NaN loss. The standard fix is gradient clipping: if the gradient norm exceeds a threshold (typically 1.0 or 5.0), the gradient is rescaled. Unlike vanishing gradients, exploding gradients are easier to detect and fix."}
KNOWLEDGE_BASE["gradient_clipping"] = {"aliases": ["gradient clipping", "clip gradients", "gradient norm clipping", "prevent exploding gradient", "max norm", "clipping threshold"], "explanation": "Gradient clipping rescales the gradient vector when its L2 norm exceeds a predefined threshold: if norm(g) > threshold, then g = g * (threshold / norm(g)). This preserves the gradient direction while controlling magnitude. It is standard in all modern RNN and transformer training pipelines."}
KNOWLEDGE_BASE["lstm"] = {"aliases": ["lstm", "long short term memory", "lstm network", "what is lstm", "lstm architecture", "lstm vs rnn"], "explanation": "LSTM (Long Short-Term Memory), introduced by Hochreiter and Schmidhuber (1997), is an RNN variant designed to learn long-range dependencies. It introduces a cell state as long-term memory and three gates (forget, input, output) using sigmoid activations. The additive cell state update allows gradients to flow without vanishing over many timesteps."}
KNOWLEDGE_BASE["cell_state"] = {"aliases": ["cell state", "lstm cell state", "c_t", "long term memory lstm", "conveyor belt lstm", "lstm memory cell"], "explanation": "The cell state c_t in an LSTM is a separate memory vector that runs through the entire sequence with only minor linear interactions, acting as a conveyor belt for long-term information. It is updated additively: c_t = f_t * c_{t-1} + i_t * g_t. This additive update allows gradients to flow back without vanishing. The cell state is filtered through the output gate to produce h_t."}
KNOWLEDGE_BASE["forget_gate"] = {"aliases": ["forget gate", "lstm forget", "what to forget lstm", "f_t gate", "sigmoid forget", "cell state reset"], "explanation": "The forget gate f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f) decides what fraction of the previous cell state to retain. Values near 1 mean keep everything, near 0 mean forget completely. For example, in language modeling, the forget gate might reset subject-verb agreement memory when a sentence ends. It was added by Gers et al. (1999)."}
KNOWLEDGE_BASE["input_gate"] = {"aliases": ["input gate", "lstm input gate", "i_t gate", "what to write lstm", "update gate lstm", "candidate cell state"], "explanation": "The input gate has two parts: i_t = sigmoid(W_i * [h_{t-1}, x_t]) controls how much of the candidate to write, and g_t = tanh(W_g * [h_{t-1}, x_t]) is the candidate cell update. The actual cell update is i_t * g_t. This gate determines what new information from the current input should be stored in the cell state."}
KNOWLEDGE_BASE["output_gate"] = {"aliases": ["output gate", "lstm output gate", "o_t gate", "what to output lstm", "hidden state from cell state", "lstm output"], "explanation": "The output gate o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o) controls what part of the cell state is exposed as the hidden state: h_t = o_t * tanh(c_t). The LSTM can maintain information in the cell state without necessarily exposing it, selectively reading from the cell state to produce output at each timestep."}
KNOWLEDGE_BASE["lstm_equations"] = {"aliases": ["lstm equations", "lstm formulas", "lstm math", "lstm forward pass", "lstm gate equations", "lstm computation"], "explanation": "Full LSTM forward pass: f_t = sigmoid(W_f[h_{t-1},x_t]+b_f), i_t = sigmoid(W_i[h_{t-1},x_t]+b_i), g_t = tanh(W_g[h_{t-1},x_t]+b_g), o_t = sigmoid(W_o[h_{t-1},x_t]+b_o), c_t = f_t*c_{t-1} + i_t*g_t, h_t = o_t*tanh(c_t). The key insight is that c_t updates additively, enabling gradient flow across many timesteps."}
KNOWLEDGE_BASE["gru"] = {"aliases": ["gru", "gated recurrent unit", "what is gru", "gru architecture", "gru vs lstm", "simplified lstm", "cho 2014"], "explanation": "GRU (Gated Recurrent Unit), proposed by Cho et al. (2014), simplifies LSTM by merging the cell state and hidden state into one, and using only two gates: reset (r_t) and update (z_t). The update gate combines LSTMs forget and input gates. GRUs have fewer parameters and train faster, often matching LSTM performance on smaller datasets."}
KNOWLEDGE_BASE["reset_gate"] = {"aliases": ["reset gate", "gru reset", "r_t gate", "gru forget", "what reset gate does", "gru previous hidden state"], "explanation": "The reset gate r_t = sigmoid(W_r * [h_{t-1}, x_t]) in a GRU controls how much of the previous hidden state to use when computing the candidate hidden state. When r_t is near 0, the unit ignores the previous state and resets. When r_t is near 1, the previous state is fully used, similar to a standard RNN."}
KNOWLEDGE_BASE["update_gate_gru"] = {"aliases": ["update gate", "gru update gate", "z_t gate", "gru interpolation", "update gate gru", "gru memory control"], "explanation": "The update gate z_t = sigmoid(W_z * [h_{t-1}, x_t]) controls how much of the previous hidden state to carry forward: h_t = (1-z_t)*h_{t-1} + z_t*h_tilde. When z_t is near 1, the new candidate dominates; when near 0, the previous hidden state is preserved unchanged, effectively skipping the current timestep for long-term dependency handling."}
KNOWLEDGE_BASE["gru_vs_lstm"] = {"aliases": ["gru vs lstm", "gru or lstm", "difference gru lstm", "when to use gru", "when to use lstm", "gru lstm comparison"], "explanation": "GRU has fewer parameters (no separate cell state, two gates vs three) making it faster to train and better on small datasets. LSTM has more expressive power due to the separate cell state and finer-grained control via three gates, often outperforming GRU on tasks requiring very long-range memory. GRU is preferred when speed matters, LSTM when maximum sequence modeling capacity is needed."}
KNOWLEDGE_BASE["sequence_to_sequence"] = {"aliases": ["seq2seq", "sequence to sequence", "encoder decoder", "machine translation rnn", "seq2seq model", "encoder decoder rnn"], "explanation": "Seq2Seq models use an encoder RNN that reads the input sequence and compresses it into a fixed-size context vector (the final hidden state), and a decoder RNN that generates output from this vector. This architecture allows variable-length input and output sequences and is the foundation for machine translation, summarization, and dialogue systems."}
KNOWLEDGE_BASE["context_vector"] = {"aliases": ["context vector", "encoder output vector", "bottleneck seq2seq", "fixed size representation", "encoder final state", "seq2seq bottleneck"], "explanation": "The context vector is the encoder's final hidden state in a seq2seq model, compressing the entire input sequence into a fixed-size vector. The bottleneck problem arises because all input information must fit in this single vector, causing information loss for long sequences. The attention mechanism was introduced to overcome this."}
KNOWLEDGE_BASE["teacher_forcing"] = {"aliases": ["teacher forcing", "ground truth input decoder", "training seq2seq", "exposure bias", "scheduled sampling", "teacher forcing training"], "explanation": "Teacher forcing feeds the ground truth token from the previous timestep as decoder input during training, rather than the model's own prediction. This speeds up convergence but causes exposure bias — a mismatch between training and inference. Scheduled sampling gradually replaces ground truth with model predictions to address this."}
KNOWLEDGE_BASE["attention_mechanism"] = {"aliases": ["attention", "attention mechanism", "bahdanau attention", "additive attention", "alignment scores", "attention weights", "context vector attention", "luong attention"], "explanation": "Attention (Bahdanau et al., 2014) allows the decoder to look at all encoder hidden states at each decoding step. Alignment scores e_{t,s} are computed for each encoder state, normalized via softmax to get attention weights alpha_{t,s}, and the context vector is their weighted sum c_t = sum(alpha * h_s). This removes the fixed-size bottleneck and improves performance on long sequences."}
KNOWLEDGE_BASE["bahdanau_vs_luong"] = {"aliases": ["bahdanau attention", "luong attention", "additive vs multiplicative attention", "dot product attention", "attention types"], "explanation": "Bahdanau (additive) attention computes alignment scores using a feed-forward network: score(h_t, h_s) = v^T * tanh(W_1*h_t + W_2*h_s). Luong (multiplicative) attention uses dot products: score = h_t^T * W * h_s. Luong attention is computationally simpler and faster. Bahdanau computes alignment before the decoder output; Luong uses it after."}
KNOWLEDGE_BASE["bidirectional_rnn"] = {"aliases": ["bidirectional rnn", "birnn", "bidir rnn", "forward backward rnn", "context both directions", "bidirectional lstm", "bilstm"], "explanation": "A Bidirectional RNN runs two separate RNNs over the input — one forward and one backward — and concatenates their hidden states at each timestep. This gives the model access to both past and future context at every position, crucial for tasks like NER and POS tagging. BiRNNs cannot be used for generation tasks since future tokens are unknown at inference."}
KNOWLEDGE_BASE["deep_rnn"] = {"aliases": ["deep rnn", "stacked rnn", "multi layer rnn", "stacked lstm", "hierarchical rnn", "deep recurrent network"], "explanation": "Deep RNNs stack multiple recurrent layers where the hidden state of one layer becomes the input to the next, with each layer learning increasingly abstract temporal representations. Stacking 2-4 LSTM layers is standard in practice. Dropout between layers is used for regularization. Beyond 4 layers, residual connections are needed."}
KNOWLEDGE_BASE["rnn_language_model"] = {"aliases": ["rnn language model", "rnn lm", "language modeling rnn", "next word prediction rnn", "rnn text generation", "character rnn"], "explanation": "An RNN language model predicts the probability of the next token given all previous tokens: P(w_t | w_1,...,w_{t-1}). The input token is embedded, fed to the RNN, and the output hidden state is projected through a softmax over the vocabulary. Training minimizes cross-entropy loss. Perplexity = exp(average cross-entropy) is the standard evaluation metric."}
KNOWLEDGE_BASE["perplexity"] = {"aliases": ["perplexity", "language model evaluation", "ppl", "rnn perplexity", "how to evaluate language model", "cross entropy language model"], "explanation": "Perplexity measures how well a language model predicts a test sequence: PPL = exp(-1/N * sum(log P(w_t|context))). Lower perplexity means the model assigns higher probability to the test data. A perplexity of k means the model is as confused as if it chose uniformly from k words. It is the standard metric for comparing language models."}
KNOWLEDGE_BASE["rnn_dropout"] = {"aliases": ["rnn dropout", "dropout lstm", "variational dropout", "recurrent dropout", "dropout rnn regularization", "how to regularize rnn"], "explanation": "Standard dropout hurts RNN training by disrupting memory across timesteps. Variational dropout by Gal and Ghahramani (2016) fixes this by applying the same dropout mask at every timestep within a sequence. Dropout should be applied only to non-recurrent connections (between layers), not within the recurrent step, unless using the variational formulation."}
KNOWLEDGE_BASE["embedding_layer"] = {"aliases": ["word embedding rnn", "embedding layer", "input embedding", "word vector rnn", "one hot vs embedding", "rnn input representation"], "explanation": "RNNs typically receive word embeddings (dense vectors) rather than raw one-hot vectors as input, reducing dimensionality drastically. Embeddings are either trained from scratch with the RNN or initialized with pretrained vectors (Word2Vec, GloVe, FastText). The embedding matrix is learned during training via backpropagation."}
KNOWLEDGE_BASE["rnn_vs_transformer"] = {"aliases": ["rnn vs transformer", "why transformers replaced rnn", "rnn limitations", "transformer better than rnn", "attention vs recurrence"], "explanation": "RNNs process sequences step-by-step making them slow to train and unable to parallelize across timesteps. Transformers use self-attention to process all positions simultaneously, enabling full parallelism and handling very long-range dependencies more directly. However, RNNs are more memory-efficient for very long sequences and still used in streaming inference settings."}
KNOWLEDGE_BASE["rnn_limitations"] = {"aliases": ["rnn limitations", "problems with rnn", "rnn disadvantages", "rnn weaknesses", "why rnn is hard to train", "rnn challenges"], "explanation": "Key RNN limitations: (1) Vanishing/exploding gradients make learning long-range dependencies difficult. (2) Sequential computation prevents parallelization, making training slow. (3) Fixed-size hidden state bottlenecks information capacity. (4) Difficult to capture hierarchical structure. LSTMs/GRUs address (1), attention addresses (3), and transformers address (2) and (4)."}

# ── LAZY MODEL STATE ───────────────────────────────────────────────────────────
_models_loaded = False
_model_lock = threading.Lock()
embedder     = None
vader        = None
tfidf        = None
tfidf_matrix = None
alias_labels = []

def load_models():
    global _models_loaded, embedder, vader, tfidf, tfidf_matrix, alias_labels
    if _models_loaded:
        return
    with _model_lock:
        if _models_loaded:  # double-checked locking
            return
        print("Loading NLP models...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        vader    = SentimentIntensityAnalyzer()

        alias_texts = []
        alias_labels = []
        for concept, data in KNOWLEDGE_BASE.items():
            for alias in data["aliases"]:
                alias_texts.append(alias)
                alias_labels.append(concept)

        tfidf        = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(alias_texts)

        for concept, data in KNOWLEDGE_BASE.items():
            data["embedding"] = embedder.encode(data["explanation"], convert_to_tensor=True)

        _models_loaded = True
        print(f"Pipeline ready. {len(KNOWLEDGE_BASE)} concepts loaded.")

# ── GROQ CLIENT ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a contrarian teaching assistant specialized in RNNs and sequential deep learning.

RULE 1 — ALWAYS DISAGREE IN TONE.
Never say "You're correct" or validate directly.
Open with pushback like: "Well, not exactly...", "That's one way to look at it, but...", "You're missing something crucial here..."

RULE 2 — TWO RESPONSE PATHS based on the NLP label provided:
- If label is CORRECT: Reluctantly acknowledge it, then immediately challenge with a deeper viewpoint or edge case they haven't considered.
- If label is INCORRECT: Firmly correct the misconception using the same angle the student approached it from.

RULE 3 — USE THE TONE SIGNAL.
If tone is 'confident', be more forceful. If 'uncertain', be sharper but nurturing. If 'neutral', keep balanced.

RULE 4 — CONVERSATIONAL MEMORY. You remember the full conversation. Refer back to prior exchanges when relevant.

RULE 5 — EDUCATION DOMAIN ONLY. If the question isn't about RNNs or NLP, say: "I only engage with RNN-related academic concepts."

RULE 6 — CONCISE. 2-4 sentences. No bullet points. Conversational but sharp."""

CORRECT_THRESHOLD = 0.68

# ── NLP PIPELINE ───────────────────────────────────────────────────────────────
def detect_intent(question):
    q_vec  = tfidf.transform([question.lower()])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    best_idx = int(np.argmax(scores))
    if scores[best_idx] > 0.15:
        return alias_labels[best_idx]
    q_emb = embedder.encode(question, convert_to_tensor=True)
    best_score, best_concept = 0.0, None
    for concept, data in KNOWLEDGE_BASE.items():
        score = float(util.cos_sim(q_emb, data["embedding"]))
        if score > best_score:
            best_score, best_concept = score, concept
    return best_concept if best_score > 0.40 else None

def evaluate_understanding(user_understanding, concept):
    gt_emb  = KNOWLEDGE_BASE[concept]["embedding"]
    usr_emb = embedder.encode(user_understanding, convert_to_tensor=True)
    similarity = float(util.cos_sim(usr_emb, gt_emb))
    return {
        "label": "CORRECT" if similarity >= CORRECT_THRESHOLD else "INCORRECT",
        "similarity": round(similarity, 4)
    }

def analyze_sentiment(text):
    scores   = vader.polarity_scores(text)
    compound = scores["compound"]
    tone = "confident" if compound >= 0.3 else ("uncertain" if compound <= -0.2 else "neutral")
    return {"compound": compound, "tone": tone}

def run_pipeline(question, user_understanding):
    concept = detect_intent(question)
    if not concept:
        return {"label": "OUT_OF_SCOPE", "concept": None, "similarity": None, "tone": None}
    sim  = evaluate_understanding(user_understanding, concept)
    sent = analyze_sentiment(user_understanding)
    return {"concept": concept, "label": sim["label"], "similarity": sim["similarity"], "tone": sent["tone"]}

# ── SESSION STORE ──────────────────────────────────────────────────────────────
sessions = {}

# ── FLASK ROUTES ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": _models_loaded})

@app.route("/chat", methods=["POST"])
def chat():
    load_models()  # lazy load — no-op after first call

    data          = request.json
    session_id    = data.get("session_id", "default")
    question      = data.get("question", "").strip()
    understanding = data.get("understanding", "").strip()

    if not question or not understanding:
        return jsonify({"error": "Both fields are required."}), 400

    result = run_pipeline(question, understanding)

    if result["label"] == "OUT_OF_SCOPE":
        return jsonify({
            "response": "I only engage with RNN-related academic concepts. Ask me something worth disagreeing about.",
            "label": "OUT_OF_SCOPE", "similarity": None, "tone": None, "concept": None
        })

    user_message = (
        f"Concept/Question: {question}\n"
        f"My understanding: {understanding}\n"
        f"[NLP: label={result['label']}, similarity={result['similarity']}, tone={result['tone']}]"
    )

    if session_id not in sessions:
        sessions[session_id] = []
    sessions[session_id].append({"role": "user", "content": user_message})

    groq_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + sessions[session_id],
        temperature=0.75,
        max_tokens=250
    )
    reply = groq_response.choices[0].message.content
    sessions[session_id].append({"role": "assistant", "content": reply})

    return jsonify({
        "response": reply,
        "label": result["label"],
        "similarity": result["similarity"],
        "tone": result["tone"],
        "concept": result["concept"]
    })

@app.route("/reset", methods=["POST"])
def reset():
    data       = request.json
    session_id = data.get("session_id", "default")
    sessions.pop(session_id, None)
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    