class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples = len(X)
        n_labels = len(set(y))

        # Stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        best_feat, best_thresh = self._best_split(X, y)
        left_idxs, right_idxs = self._split(X, best_feat, best_thresh)
        left = self._grow_tree([X[i] for i in left_idxs], [y[i] for i in left_idxs], depth + 1)
        right = self._grow_tree([X[i] for i in right_idxs], [y[i] for i in right_idxs], depth + 1)
        return self.Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y):
        best_gini = float('inf')
        split_idx, split_thresh = None, None
        for feat_idx in range(len(X[0])):
            X_column = [x[feat_idx] for x in X]
            thresholds = set(X_column)
            for threshold in thresholds:
                gini = self._gini_index(y, X_column, threshold)
                if gini < best_gini:
                    best_gini = gini
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _gini_index(self, y, X_column, threshold):
        left_idxs = [i for i in range(len(X_column)) if X_column[i] <= threshold]
        right_idxs = [i for i in range(len(X_column)) if X_column[i] > threshold]

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return float('inf')

        left_gini = self._gini(y, left_idxs)
        right_gini = self._gini(y, right_idxs)
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        gini = (n_l / n) * left_gini + (n_r / n) * right_gini
        return gini

    def _gini(self, y, idxs):
        class_counts = {}
        for idx in idxs:
            label = y[idx]
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        impurity = 1
        for label in class_counts:
            prob_of_lbl = class_counts[label] / len(idxs)
            impurity -= prob_of_lbl ** 2
        return impurity

    def _split(self, X, split_idx, split_thresh):
        left_idxs = [i for i in range(len(X)) if X[i][split_idx] <= split_thresh]
        right_idxs = [i for i in range(len(X)) if X[i][split_idx] > split_thresh]
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        counter = {}
        for label in y:
            if label not in counter:
                counter[label] = 0
            counter[label] += 1
        most_common = max(counter, key=counter.get)
        return most_common

    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)