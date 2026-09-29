import unittest

import numpy as np
import torch

from control.signal_lamp_state import LampStateClassifier


class ConstantModel:
    def __init__(self, state_index):
        self.state_index = state_index

    def __call__(self, batch):
        logits = torch.zeros((len(batch), 4), device=batch.device)
        logits[:, self.state_index] = 12
        return logits


class LampStateEvidenceTests(unittest.TestCase):
    def classify(self, predicted, color):
        classifier = LampStateClassifier.__new__(LampStateClassifier)
        classifier.device = 'cpu'
        classifier.threshold = .9
        classifier.model = ConstantModel(predicted)
        image = np.full((64, 48, 3), color, dtype=np.uint8)
        return classifier.predict([image])[0]

    def test_red_prediction_requires_red_pixels(self):
        self.assertEqual(self.classify(1, [255, 255, 0])['state'], 'UNKNOWN')
        self.assertEqual(self.classify(1, [255, 0, 0])['state'], 'RED')

    def test_yellow_prediction_requires_yellow_pixels(self):
        self.assertEqual(self.classify(2, [255, 0, 0])['state'], 'UNKNOWN')
        self.assertEqual(self.classify(2, [255, 255, 0])['state'], 'YELLOW')


if __name__ == '__main__':
    unittest.main()
