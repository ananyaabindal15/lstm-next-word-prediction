import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Shakespearean Next Word Predictor",
    page_icon="📖",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📖 Next Word Prediction with LSTM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Trained on Shakespeare\'s <i>Hamlet</i> using Deep Recurrent Neural Networks</div>', unsafe_allow_html=True)

# 1. Cached Asset Loaders
@st.cache_resource
def load_assets():
    models = {}
    if os.path.exists("next_word_lstm.h5"):
        models["LSTM Baseline (Best Performer)"] = load_model("next_word_lstm.h5")
    if os.path.exists("next_word_lstm_model_with_early_stopping.h5"):
        models["LSTM with Early Stopping"] = load_model("next_word_lstm_model_with_early_stopping.h5")
    
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
        
    return models, tokenizer

try:
    models, tokenizer = load_assets()
    idx_to_word = {index: word for word, index in tokenizer.word_index.items()}
except Exception as e:
    st.error(f"Error loading models or tokenizer: {e}")
    st.stop()

# 2. Sidebar Settings
with st.sidebar:
    st.header("⚙️ Model Configuration")
    selected_model_name = st.selectbox("Select Model Checkpoint:", list(models.keys()))
    selected_model = models[selected_model_name]
    max_sequence_len = selected_model.input_shape[1] + 1
    
    st.markdown("---")
    st.header("🎛️ Generation Controls")
    num_suggestions = st.slider("Number of Top Candidates (Top-K):", min_value=1, max_value=10, value=5)
    generate_words_count = st.slider("Generate Sequence (Multi-word rollout):", min_value=1, max_value=15, value=1)
    temperature = st.slider("Sampling Temperature:", min_value=0.1, max_value=1.5, value=0.7, step=0.1,
                            help="Lower values make predictions more deterministic; higher values introduce variety.")

    st.markdown("---")
    st.caption("Vocabulary Size: **4,818 words**")
    st.caption(f"Context Window: **{max_sequence_len - 1} tokens**")

# Helper Functions
def get_next_word_probs(text, model, tokenizer, max_seq_len, temp=1.0):
    tokens = tokenizer.texts_to_sequences([text])[0]
    if len(tokens) >= max_seq_len:
        tokens = tokens[-(max_seq_len - 1):]
    padded = pad_sequences([tokens], maxlen=max_seq_len - 1, padding='pre')
    
    raw_preds = model.predict(padded, verbose=0)[0]
    
    # Temperature scaling
    if temp != 1.0 and temp > 0:
        logits = np.log(np.maximum(raw_preds, 1e-10)) / temp
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
    else:
        probs = raw_preds
        
    return probs

def generate_text_sequence(seed_text, n_words, model, tokenizer, max_seq_len, temp=0.7):
    curr_text = seed_text
    generated = []
    for _ in range(n_words):
        probs = get_next_word_probs(curr_text, model, tokenizer, max_seq_len, temp)
        # Sample with probability distribution
        chosen_idx = np.random.choice(len(probs), p=probs)
        chosen_word = idx_to_word.get(chosen_idx, "")
        if not chosen_word:
            break
        generated.append(chosen_word)
        curr_text += " " + chosen_word
    return " ".join(generated)

# 3. Main Interface
col_input, col_preset = st.columns([3, 1])

with col_preset:
    st.write("**Sample Prompts:**")
    if st.button("🎭 'To be or not to'"):
        st.session_state["prompt_input"] = "To be or not to"
    if st.button("👑 'Shall I compare thee to a'"):
        st.session_state["prompt_input"] = "Shall I compare thee to a"
    if st.button("⚔️ 'Barn. Last night of all, When yond same'"):
        st.session_state["prompt_input"] = "Barn. Last night of all, When yond same"

default_text = st.session_state.get("prompt_input", "To be or not to")

with col_input:
    input_text = st.text_input("Enter your Shakespearean phrase:", value=default_text)

predict_btn = st.button("⚡ Predict Next Word", type="primary", use_container_width=True)

if predict_btn or input_text:
    if not input_text.strip():
        st.warning("Please enter some text to predict.")
    else:
        with st.spinner("Analyzing sequence..."):
            probs = get_next_word_probs(input_text, selected_model, tokenizer, max_sequence_len, temp=temperature)
            top_indices = np.argsort(probs)[-num_suggestions:][::-1]
            top_candidates = [(idx_to_word.get(idx, "<unk>"), float(probs[idx])) for idx in top_indices]

        # Display Top Choice
        best_word, best_prob = top_candidates[0]
        st.success(f"**Top Predicted Next Word:** `{best_word}` (Confidence: {best_prob*100:.2f}%)")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader(f"Top {num_suggestions} Predictions")
            chart_data = {cand[0]: cand[1] * 100 for cand in reversed(top_candidates)}
            st.bar_chart(chart_data)

        with col2:
            st.subheader("Candidate Breakdown")
            table_data = [
                {"Rank": f"#{i+1}", "Candidate Word": word, "Probability": f"{prob*100:.2f}%"}
                for i, (word, prob) in enumerate(top_candidates)
            ]
            st.dataframe(table_data, use_container_width=True, hide_index=True)

        if generate_words_count > 1:
            st.markdown("---")
            st.subheader(f"✨ Multi-Word Continuation ({generate_words_count} words)")
            full_continuation = generate_text_sequence(
                input_text, generate_words_count, selected_model, tokenizer, max_sequence_len, temp=temperature
            )
            st.info(f"**Prompt:** *{input_text}* \n\n**Generated Continuation:** **{input_text} {full_continuation}**")

# 4. Evaluation Metrics Section
st.markdown("---")
with st.expander("📊 View Model Benchmark & Evaluation Metrics"):
    if os.path.exists("evaluation_metrics.png"):
        st.image("evaluation_metrics.png", caption="Model Benchmark Comparison on Hamlet Test Set")
    
    st.markdown("""
    ### Test Set Benchmark (5,147 samples, 80/20 split)
    | Model Architecture | Cross-Entropy Loss | Perplexity (PPL) | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | Top-10 Accuracy |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
    | **LSTM Baseline (`next_word_lstm.h5`)** | **3.7017** | **40.51** | **38.61%** | **51.29%** | **58.05%** | **66.54%** |
    | **LSTM Early Stopping (`early_stopping.h5`)** | **4.8181** | **123.73** | **14.65%** | **26.35%** | **33.48%** | **44.82%** |
    """)
