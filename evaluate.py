import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

print("==================================================")
print("     LSTM Next Word Prediction - Model Evaluation")
print("==================================================")

# 1. Load Tokenizer
print("\n[1/5] Loading tokenizer and dataset...")
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

vocab_size = len(tokenizer.word_index) + 1
print(f"-> Vocabulary Size: {vocab_size}")

with open("hamlet.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# 2. Build N-gram sequences
print("\n[2/5] Generating N-gram sequences from hamlet.txt...")
input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = 14
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)

x = input_sequences[:, :-1]
y = input_sequences[:, -1]
total_samples = len(x)
print(f"-> Total N-gram sequence samples: {total_samples}")

# Train-test split (80-20, fixed seed for consistency)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
print(f"-> Evaluation test set samples: {len(x_test)}")

# 3. Evaluation Function
models_to_evaluate = [
    {
        "name": "LSTM Baseline (next_word_lstm.h5)",
        "path": "next_word_lstm.h5",
        "key": "lstm_baseline"
    },
    {
        "name": "LSTM Early Stopping (next_word_lstm_model_with_early_stopping.h5)",
        "path": "next_word_lstm_model_with_early_stopping.h5",
        "key": "lstm_early_stopping"
    }
]

def evaluate_model(model_path, x_data, y_data, batch_size=256):
    print(f"\nEvaluating: {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    
    # Calculate predictions in batches
    num_samples = len(x_data)
    total_loss = 0.0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    
    cce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    for i in range(0, num_samples, batch_size):
        bx = x_data[i:i + batch_size]
        by = y_data[i:i + batch_size]
        
        preds = model.predict(bx, verbose=0)  # shape: (B, vocab_size)
        loss = cce_loss_fn(by, preds).numpy()
        total_loss += loss * len(bx)
        
        # Sort predictions
        top_k_indices = np.argsort(preds, axis=1)[:, ::-1]
        
        top1_correct += np.sum(top_k_indices[:, 0] == by)
        top3_correct += np.sum([by[j] in top_k_indices[j, :3] for j in range(len(by))])
        top5_correct += np.sum([by[j] in top_k_indices[j, :5] for j in range(len(by))])
        top10_correct += np.sum([by[j] in top_k_indices[j, :10] for j in range(len(by))])
        
    avg_loss = total_loss / num_samples
    perplexity = np.exp(avg_loss)
    top1_acc = (top1_correct / num_samples) * 100.0
    top3_acc = (top3_correct / num_samples) * 100.0
    top5_acc = (top5_correct / num_samples) * 100.0
    top10_acc = (top10_correct / num_samples) * 100.0
    
    return {
        "loss": round(float(avg_loss), 4),
        "perplexity": round(float(perplexity), 2),
        "top1_accuracy": round(float(top1_acc), 2),
        "top3_accuracy": round(float(top3_acc), 2),
        "top5_accuracy": round(float(top5_acc), 2),
        "top10_accuracy": round(float(top10_acc), 2)
    }

# 4. Run Evaluation
print("\n[3/5] Running quantitative evaluation on test set...")
results = {}
for m in models_to_evaluate:
    if os.path.exists(m["path"]):
        res = evaluate_model(m["path"], x_test, y_test)
        results[m["key"]] = {
            "name": m["name"],
            "metrics": res
        }
        print(f"Results for {m['name']}:")
        print(f"  - Cross-Entropy Loss: {res['loss']}")
        print(f"  - Perplexity (PPL):   {res['perplexity']}")
        print(f"  - Top-1 Accuracy:     {res['top1_accuracy']}%")
        print(f"  - Top-3 Accuracy:     {res['top3_accuracy']}%")
        print(f"  - Top-5 Accuracy:     {res['top5_accuracy']}%")
        print(f"  - Top-10 Accuracy:    {res['top10_accuracy']}%")

# Save results to json
with open("evaluation_metrics.json", "w") as f:
    json.dump(results, f, indent=4)
print("\n[4/5] Metrics saved to evaluation_metrics.json")

# 5. Generate Visualizations
print("\n[5/5] Generating metric comparison plots...")
model_labels = [v["name"].split(" (")[0] for v in results.values()]
top1_vals = [v["metrics"]["top1_accuracy"] for v in results.values()]
top3_vals = [v["metrics"]["top3_accuracy"] for v in results.values()]
top5_vals = [v["metrics"]["top5_accuracy"] for v in results.values()]
ppl_vals = [v["metrics"]["perplexity"] for v in results.values()]
loss_vals = [v["metrics"]["loss"] for v in results.values()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Accuracies
x_indices = np.arange(len(model_labels))
bar_width = 0.25

ax1.bar(x_indices - bar_width, top1_vals, width=bar_width, label="Top-1 Acc (%)", color="#4285F4")
ax1.bar(x_indices, top3_vals, width=bar_width, label="Top-3 Acc (%)", color="#34A853")
ax1.bar(x_indices + bar_width, top5_vals, width=bar_width, label="Top-5 Acc (%)", color="#FBBC05")
ax1.set_title("Next Word Prediction Accuracy (Higher is better)", fontsize=13, fontweight='bold')
ax1.set_xticks(x_indices)
ax1.set_xticklabels(model_labels, fontsize=10)
ax1.set_ylabel("Accuracy (%)")
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.6)

for i in range(len(model_labels)):
    ax1.text(i - bar_width, top1_vals[i] + 0.5, f"{top1_vals[i]}%", ha='center', fontsize=9)
    ax1.text(i, top3_vals[i] + 0.5, f"{top3_vals[i]}%", ha='center', fontsize=9)
    ax1.text(i + bar_width, top5_vals[i] + 0.5, f"{top5_vals[i]}%", ha='center', fontsize=9)

# Subplot 2: Perplexity & Loss
color1 = '#EA4335'
color2 = '#9C27B0'
ax2_twin = ax2.twinx()

b1 = ax2.bar(x_indices - 0.15, ppl_vals, width=0.3, label="Perplexity (PPL)", color=color1)
b2 = ax2_twin.bar(x_indices + 0.15, loss_vals, width=0.3, label="Test Loss", color=color2)

ax2.set_title("Perplexity & Test Loss (Lower is better)", fontsize=13, fontweight='bold')
ax2.set_xticks(x_indices)
ax2.set_xticklabels(model_labels, fontsize=10)
ax2.set_ylabel("Perplexity", color=color1)
ax2_twin.set_ylabel("Loss", color=color2)
ax2.tick_params(axis='y', labelcolor=color1)
ax2_twin.tick_params(axis='y', labelcolor=color2)
ax2.grid(axis='y', linestyle='--', alpha=0.6)

for i in range(len(model_labels)):
    ax2.text(i - 0.15, ppl_vals[i] + 5, f"{ppl_vals[i]}", ha='center', color=color1, fontweight='bold', fontsize=9)
    ax2_twin.text(i + 0.15, loss_vals[i] + 0.1, f"{loss_vals[i]}", ha='center', color=color2, fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig("evaluation_metrics.png", dpi=300)
print("-> Plot saved to evaluation_metrics.png")
print("\nEvaluation pipeline successfully finished!")
