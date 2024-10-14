import numpy as np


class Statistics:
    def __init__(self):
        self.mean = 0
        self.confidences = []

    def compute_mean(self, c):
        if c > 0:
            self.confidences.append(c)
            self.mean = sum(self.confidences)/len(self.confidences)
        else:
            self.mean = sum(self.confidences)/len(self.confidences)

    def get_mean(self):
        return self.mean

    def get_list(self):
        return self.confidences
