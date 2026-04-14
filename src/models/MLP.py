from sklearn.neural_network import MLPClassifier


class MLP(MLPClassifier):
    """Sklearn MLP with a DeepOD-compatible decision_function interface.

    Accepts and ignores DeepOD-style kwargs (random_state, device, verbose)
    so it can be used as a drop-in replacement for DeepSAD.
    """

    def __init__(self, *args, random_state=None, device=None, verbose=None, **kwargs):
        # Pass random_state to sklearn; ignore device/verbose (sklearn-only)
        init_kwargs = kwargs.copy()
        if random_state is not None:
            init_kwargs["random_state"] = random_state
        super().__init__(*args, **init_kwargs)

    def decision_function(self, X):
        proba = self.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba.ravel()


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