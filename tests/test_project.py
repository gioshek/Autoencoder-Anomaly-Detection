import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import (
    TrainingPatches, aggregate, assemble_patches, grid, masked_errors, split,
)
from src.evaluate import evaluate, metrics
from src.model import Autoencoder
from src.train import calibration_threshold, predict, seed_all
from src.visualization import evaluation_plots, history_plot, reconstruction_from_patches


class ProjectTests(unittest.TestCase):
    def setUp(self):
        # Проверки используют временные искусственные данные без обучения и официального теста.
        seed_all()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save_image(self, path, seed=7, shape=(170, 300, 3)):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
        Image.fromarray(image).save(path)
        return image

    def test_split(self):
        for i in range(20):
            self.save_image(self.root / 'train/good' / f'{i:03d}.png', seed=i, shape=(128, 128, 3))
        manifest = split(self.root)
        self.assertEqual(manifest, split(self.root))
        groups = [set(manifest[key]) for key in ['training', 'validation', 'calibration']]
        self.assertEqual([len(group) for group in groups], [14, 3, 3])
        self.assertEqual(len(set.union(*groups)), 20)
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        patches = TrainingPatches(self.root, manifest['training'])
        self.assertEqual(patches[0].shape, (3, 128, 128))

    def test_grid_and_masked_mse(self):
        path = self.root / 'wide.png'
        image = self.save_image(path)
        patches, mask, shape = grid(path)
        self.assertEqual(shape, (256, 384))
        self.assertEqual(mask.sum().item(), 170 * 300)
        expected = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)
        self.assertTrue(torch.equal(assemble_patches(patches, shape)[:, :170, :300], expected))
        self.assertTrue(torch.equal(masked_errors(patches, patches, mask), torch.zeros(6)))
        self.assertTrue(torch.equal(
            masked_errors(torch.zeros_like(patches), torch.ones_like(patches), mask), torch.ones(6),
        ))
        padded = patches.clone()
        padded += (1 - mask) * 100
        self.assertTrue(torch.equal(masked_errors(patches, padded, mask), torch.zeros(6)))

    def test_model_and_prediction(self):
        model = Autoencoder().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 23603)
        with torch.inference_mode():
            x = torch.zeros(1, 3, 128, 128)
            self.assertEqual(model.encoder(x).shape, (1, 32, 8, 8))
            self.assertEqual(model(x).shape, x.shape)
        path = self.root / 'image.png'
        self.save_image(path)
        scores, x, y, mask, _ = predict(model, path)
        errors = masked_errors(x, y, mask).numpy()
        self.assertEqual(scores, aggregate(errors, mask.sum((1, 2, 3)).numpy()))

    def test_threshold_and_aggregation(self):
        self.assertAlmostEqual(calibration_threshold(np.arange(1, 21)), 19.05)
        self.assertEqual(aggregate(np.arange(9), np.ones(9)), {'global_mse': 4, 'local_patch': 7})
        self.assertEqual(aggregate(np.arange(49), np.ones(49))['local_patch'], 42)
        self.assertAlmostEqual(aggregate([1, 3], [1, 3])['global_mse'], 2.5)

    def test_metrics(self):
        result = metrics([0, 0, 1, 1], [.1, .4, .3, .9], .3, replicates=50)
        self.assertEqual(result['roc_auc'], .75)
        self.assertEqual(result['confusion_matrix'], [[1, 1], [1, 1]])
        for key in ['precision', 'recall', 'f1', 'false_positive_rate']:
            self.assertEqual(result[key], .5)

    def test_plots(self):
        path = self.root / 'image.png'
        self.save_image(path)
        _, x, y, mask, shape = predict(Autoencoder().eval(), path)
        reconstruction_from_patches(x, y, mask, shape, self.root / 'reconstruction.png')
        history_plot([{'epoch': 1, 'training_mse': .1, 'validation_mse': .2}], self.root / 'history.png')
        values = metrics([0, 0, 1, 1], [.1, .4, .3, .9], .3, replicates=10)
        methods = ['global_mse', 'local_patch']
        rows = [{'label': label, 'global_mse': score, 'local_patch': score}
                for label, score in zip([0, 0, 1, 1], [.1, .4, .3, .9])]
        evaluation_plots(rows, {m: values for m in methods}, {m: .3 for m in methods}, self.root)
        self.assertEqual(len(list(self.root.glob('*.png'))), 9)

    def test_evaluation_requires_confirmation(self):
        with self.assertRaisesRegex(ValueError, 'confirm-final'):
            evaluate(self.root, self.root, confirmed=False)


if __name__ == '__main__':
    unittest.main()
