import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP:
    """PyTorch MLP binary classifier with DeepOD-compatible interface.

    Drop-in replacement for DeepSAD: same fit(X, y) / decision_function(X) API.
    Runs on GPU when device='cuda'.
    """

    def __init__(
        self,
        hidden_dims=(128, 64),
        lr=1e-3,
        epochs=50,
        batch_size=256,
        random_state=42,
        device="cpu",
        verbose=0,
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self._model = None

    def _build_model(self, input_dim):
        torch.manual_seed(self.random_state)
        layers = []
        prev = input_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X, y):
        self._model = self._build_model(X.shape[1])
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._model(xb).squeeze(), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1}/{self.epochs} loss={total_loss/len(loader):.4f}")
        return self

    def decision_function(self, X):
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            scores = self._model(X_t).squeeze().cpu().numpy()
        return scores


class MLPTF:
    """
    TensorFlow/Keras MLP with a DeepOD-compatible interface (fit / decision_function).
    Used alongside DevNet and DeepSAD in semi-supervised anomaly detection experiments.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=(128, 64),
        lr=1e-3,
        epochs=20,
        batch_size=256,
        verbose=0,
        random_state=42,
    ):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(random_state)

        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(input_dim,)))

        for h in hidden_dims:
            self.model.add(tf.keras.layers.Dense(h, activation="relu"))

        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
        )

    def fit(self, X, y):
        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

    def decision_function(self, X):
        return self.model.predict(X, batch_size=1024).ravel()