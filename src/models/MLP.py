from sklearn.neural_network import MLPClassifier

class MLP(MLPClassifier):
    def decision_function(self, X):
        proba = self.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba.ravel()