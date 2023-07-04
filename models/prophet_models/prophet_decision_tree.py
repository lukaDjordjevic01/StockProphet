import numpy as np


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        """
        This class creates a new node for the decision tree. It stores the index of the feature for the split,
        the threshold value for the split, the left and right child nodes after the split, the reduction in variance
        due to the split, and the predicted value if it's a leaf node.

        :param feature_index:
        :param threshold:
        :param left:
        :param right:
        :param var_red:
        :param value:
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        self.value = value


class MyDecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=2):
        """
        This class represents the decision tree model. It is initialized with minimum number of samples required
        to allow a split (min_samples_split), and the maximum depth of the tree (max_depth).

        :param min_samples_split:
        :param max_depth:
        """
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        """
        This is a recursive function that splits the dataset on the feature that results in the
        largest variance reduction.It continues to build the tree by calling itself on the left
        and right child nodes, until the stopping criteria are met.

        :param dataset:
        :param curr_depth:
        :return:
        """
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["var_red"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["var_red"])

        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        """
        This function iterates over all features and all possible thresholds for each feature to find the split
        that results in the highest variance reduction. It returns the feature index, threshold, left and right datasets
        after the split, and the variance reduction due to the split.

        :param dataset:
        :param num_samples:
        :param num_features:
        :return:
        """
        best_split = {}
        max_var_red = -float("inf")
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    if curr_var_red>max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
        return best_split

    def split(self, dataset, feature_index, threshold):
        """
        This function splits the dataset into left and right nodes based on the given feature index and threshold.
        It returns the left and right datasets after the split.

        :param dataset:
        :param feature_index:
        :param threshold:
        :return:
        """
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right

    def variance_reduction(self, parent, l_child, r_child):
        """
        This function calculates the reduction in variance due to a split. It's the difference between the variance
        of the target values in the parent node and the weighted variance of the target values in the left and
        right child nodes.

        :param parent:
        :param l_child:
        :param r_child:
        :return:
        """
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l*np.var(l_child) + weight_r*np.var(r_child))
        return reduction

    def calculate_leaf_value(self, Y):
        """
        This function calculates the mean of the target values for the instances in a leaf node. This mean value
        will be returned as the prediction for these instances.

        :param Y:
        :return:
        """
        val = np.mean(Y)
        return val

    def print_tree(self, tree=None, indent=" "):
        """
        This function prints the structure of the decision tree in a format that's easy to read. It uses recursion
        to print the feature index, threshold and variance reduction at each node, and the predicted value
        at each leaf node.

        :param tree:
        :param indent:
        :return:
        """
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        """
        This function starts the training of the decision tree model. It concatenates the features and target into a
        single dataset and calls the build_tree function to construct the decision tree.

        :param X:
        :param Y:
        :return:
        """
        Y = np.reshape(Y, (-1, 1))
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def make_prediction(self, x, tree):
        """
        This is a recursive function used to traverse the decision tree based on the values of the features in the input
        instance, and return the predicted value when a leaf node is reached.

        :param x:
        :param tree:
        :return:
        """
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def predict(self, X):
        """
        This function uses the make_prediction function to predict the target value for each instance in the input
        dataset. It returns a list of predictions for all instances.

        :param X:
        :return:
        """
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
