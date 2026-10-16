import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Loading models and tokenizer...")
model1 = load_model("next_word_lstm.h5")
model2 = load_model("next_word_lstm_model_with_early_stopping.h5")

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

vocab_size = len(tokenizer.word_index) + 1
max_seq_len = model1.input_shape[1] + 1

print(f"Vocab size: {vocab_size}")
print(f"Model 1 input shape: {model1.input_shape}, Model 2 input shape: {model2.input_shape}")
print(f"Max sequence length: {max_seq_len}")

def predict_next(model, text, top_k=3):
    tokens = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences([tokens], maxlen=max_seq_len - 1, padding="pre")
    probs = model.predict(padded, verbose=0)[0]
    top_indices = np.argsort(probs)[-top_k:][::-1]
    
    idx_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
    return [(idx_to_word.get(idx, "<unk>"), float(probs[idx])) for idx in top_indices]

test_prompts = [
    "To be or not to",
    "Shall I compare thee to a",
    "Barn. Last night of all, When yond same"
]

print("\n--- Model 1 (next_word_lstm.h5) Predictions ---")
for prompt in test_prompts:
    res = predict_next(model1, prompt, top_k=3)
    print(f"Prompt: '{prompt}' -> Top predictions: {res}")

print("\n--- Model 2 (early_stopping.h5) Predictions ---")
for prompt in test_prompts:
    res = predict_next(model2, prompt, top_k=3)
    print(f"Prompt: '{prompt}' -> Top predictions: {res}")

print("\nModels and tokenizer verified successfully!")
