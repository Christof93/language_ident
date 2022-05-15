import numpy as np
from tqdm import tqdm
from collections import defaultdict
from features import sparse_vector

class MNB_classfier():
    def __init__(self):
        self.freqs = []
        self._alpha = 1

    def train(self, vectors: list, targets: list) -> None:
        assert(len(vectors) == len(targets))
        self._classes = list(set(targets))
        self._class_counts = {cl:targets.count(cl) for cl in self._classes}

        self._feature_freqs = [np.ones(vectors[0].size) for _ in self._classes]
        print(f'training classifier for {len(self._classes)} classes with {vectors[0].size} features.')
        for x, t in tqdm(zip(vectors, targets), total = len(targets)):
            cl_index = self._classes.index(t)
            for idx, val in x:
                self._feature_freqs[cl_index][idx] += val
            
        self._feature_probs = []
        for class_feature_freqs in self._feature_freqs:
            total_class_ngrams = np.sum(class_feature_freqs)
            ### p(x_i|c_k) = 1 + count(x_i, c_k) / (sum(count(x_i, c_k)) + (len(x) * alpha)) <---- Laplace smoothing
            class_feature_probs = class_feature_freqs/(total_class_ngrams + (len(class_feature_freqs)* self._alpha))
            self._feature_probs.append(class_feature_probs)

        self._class_probs = np.array([self._class_counts[cl]/len(targets) for cl in self._classes])
        print('training finished!')

    def classify(self, vector: sparse_vector) -> str:
        log_probs = np.zeros(len(self._classes))
        for cl in self._classes:
            cl_index = self._classes.index(cl)
            #log_prob = np.log(self._class_probs[cl_index]) + np.sum(vector * np.log(self._feature_probs[cl_index]))
            log_prob = np.log(self._class_probs[cl_index]) 
            for feature_index, val in vector:
                log_prob += val * np.log(self._feature_probs[cl_index][feature_index])
            log_probs[cl_index] = log_prob
        max_index = np.argmax(log_probs)
        return self._classes[max_index]
    
    def predict(self, vectors: list) -> list:
        predictions = []
        for svector in tqdm(vectors):
            pred = self.classify(svector)
            predictions.append(pred)
        return predictions


