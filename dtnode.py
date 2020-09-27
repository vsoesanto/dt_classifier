class DTNode:

    def __init__(self, feature, path, left, right):
        self.feature = feature
        self.path = path
        self.left = left
        self.right = right
        self.distribution = dict()
        self.total_docs = 0
