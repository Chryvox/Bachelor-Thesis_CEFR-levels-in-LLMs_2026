"""
Script 2: Training a probe (with sklearn.neural_network)
=======================================================
Loads a .pt file and trains an MLP (sklearn) on hidden state -> CEFR level.

Advantages of sklearn:
  - Less code (no manual training loop)
  - Built-in early stopping (prevents overfitting automatically)
  - Easier to debug

Disadvantages:
  - No GPU support (runs on CPU)
  - Less control over the optimizer (default: Adam)
"""

import argparse
import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

# ─────────────────────────────────────────────
# Training & Validation
# ─────────────────────────────────────────────

def train_sklearn_mlp(X_train, y_train, X_val, y_val, hidden_layers, epochs, lr, random_state=42):
    """
    Trains an sklearn MLP with early stopping.
    """
    # sklearn needs hidden_layer_sizes as a tuple
    # e.g. [256] -> (256,) or [512, 256] -> (512, 256)
    hidden_layer_sizes = tuple(hidden_layers)
    
    print(f"\nArchitecture: {X_train.shape[1]} -> {hidden_layer_sizes} -> {len(np.unique(y_train))} classes")
    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
    
    # MLPClassifier initialization
    # early_stopping=True: stops training when val loss doesn't improve
    # validation_fraction=0.1: uses 10% of the training data for early stopping (internally)
    # max_iter=epochs: maximum epochs (often reached earlier due to early stopping)
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=lr,
        max_iter=epochs,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,  # stop if 10 epochs show no improvement
        random_state=random_state,
        alpha=0.001,
        verbose=True  # shows training progress
    )

    # Start training
    model.fit(X_train, y_train)
    
    # Evaluation on validation data
    y_pred_val = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)
    
    # sklearn computes log_loss when we use predict_proba
    y_proba_val = model.predict_proba(X_val)
    val_log_loss = log_loss(y_val, y_proba_val)

    print(f"\n--- Result ---")
    print(f"Epochs trained: {model.n_iter_}")
    print(f"Val accuracy: {val_acc:.2%}")
    print(f"Val log loss: {val_log_loss:.4f}")
    
    return model

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states",  required=True, help="Path to the .pt file from extract_states.py")
    parser.add_argument("--layers",  type=int, nargs="+", default=[256], help="Hidden layers, e.g. --layers 512 256")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--out",     required=True, help="Output path for the trained probe model")
    args = parser.parse_args()

    # ── Load data ──
    print(f"Loading states: {args.states}")
    data = torch.load(args.states, weights_only=False)

    states      = data["states"]
    cefr_levels = data["cefr_levels"]
    languages   = data["languages"]

    if states.dim() != 2:
        raise ValueError(f"Expected states with shape [N, hidden_dim], got: {tuple(states.shape)}.")

    # Encode labels
    unique_levels = sorted(set(cefr_levels))
    label2id = {lvl: i for i, lvl in enumerate(unique_levels)}
    id2label = {i: lvl for lvl, i in label2id.items()}
    y = torch.tensor([label2id[lvl] for lvl in cefr_levels])

    print(f"Loaded: {states.shape[0]} examples, hidden_dim={states.shape[1]}")
    print(f"Languages: {sorted(set(languages))}")
    print(f"CEFR levels: {unique_levels}")

    # ── Data splitting (90% train, 10% val) ──
    n_samples = states.shape[0]
    indices = torch.randperm(n_samples)
    n_val = int(0.1 * n_samples)
    
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    # Convert to numpy (sklearn expects numpy arrays)
    X_train = states[train_idx].numpy()
    X_val   = states[val_idx].numpy()
    y_train = y[train_idx].numpy()
    y_val   = y[val_idx].numpy()

    print(f"\nSplitting: Train={X_train.shape[0]} | Val={X_val.shape[0]}")

    # ── Training ──
    print(f"\nTraining MLP (sklearn)...")
    print(f"Hyperparams: Max iter={args.epochs}, lr={args.lr}")
    
    probe = train_sklearn_mlp(
        X_train, y_train, X_val, y_val,
        hidden_layers=args.layers,
        epochs=args.epochs,
        lr=args.lr,
    )

    # ── Saving ──
    print(f"\nSaving model...")
    torch.save({
        "probe_state":           probe,
        "input_dim":             X_train.shape[1],
        "hidden_layers":         args.layers,
        "label2id":              label2id,
        "id2label":              id2label,
        "layer":                 data.get("layer"),
        "pooling":               data.get("pooling"),
        "model_id":              data.get("model_id"),
        "trained_on_languages":  sorted(set(languages)),
        "n_train_samples":       X_train.shape[0],
        "n_val_samples":         X_val.shape[0],
    }, args.out)

    print(f"Probe saved: {args.out}")


if __name__ == "__main__":
    main()