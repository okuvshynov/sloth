

class Sketch:
    def __init__(self):
        self.root = []
        self.current = []

    def get_candidates(self, sequence):
        # 1. we need to traverse the current tree to find the node we are at. 
        #    if the node doesn't exist, create it and make new root
        # 2. check if we have enough candidates in the subtree. If yes, return the current 
        #    subtree. If not, keep producing new candidates in the new tree only.
        pass

    def search_loop(self):
        # 1. pick sequences to evaluate based on current tree
        # 2. evaluate with the model
        # 3. expand the tree
        # 4. check if we have outstanding query and modify root if needed
        # 5. go to 1.
        pass

