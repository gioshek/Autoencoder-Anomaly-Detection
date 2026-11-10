# Обучение, выбор эпохи на Validation и калибровка на нормальных изображениях.
import json
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import sklearn
import torch
from PIL import __version__ as pillow_version
from torch.utils.data import DataLoader

from .data import (
    TrainingPatches, aggregate, digest, grid, masked_errors, save_json, split,
)
from .model import Autoencoder
from .reporting import write_training_report
from .visualization import history_plot, reconstruction_from_patches


def seed_all(seed=42, threads=2):
    # Один seed задаёт инициализацию весов и последовательность аугментаций.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def predict(model, path, batch_size=16):
    # Оценки изображения и фрагменты для визуализации восстановления.
    x, mask, shape = grid(path)
    reconstructed = []
    # Небольшие партии фрагментов ограничивают расход памяти при восстановлении.
    with torch.inference_mode():
        for batch in x.split(batch_size):
            reconstructed.append(model(batch))
    y = torch.cat(reconstructed)
    errors = masked_errors(x, y, mask).numpy()
    scores = aggregate(errors, mask.sum((1, 2, 3)).numpy())
    return scores, x, y, mask, shape


def calibration_threshold(scores):
    # 95-й процентиль нормальных Calibration с линейной интерполяцией.
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Пустые или нечисловые калибровочные оценки')
    return float(np.quantile(scores, .95, method='linear'))


def atomic_torch_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def make_config(root, epochs, threads):
    return {
        'seed': 42, 'epochs_max': epochs, 'patience': 8,
        'learning_rate': .001, 'optimizer': 'Adam', 'loss': 'MSE',
        'batch_size': 16, 'patch_size': 128, 'patches_per_training_image': 8,
        'preprocessing': 'native RGB; reflect pad bottom/right to multiple of 128; mask padding',
        'augmentation': 'training only: random quarter turns and horizontal flip p=0.5',
        'training_cache_limit_mib': 512,
        'architecture': 'Conv 3-8-16-24-32; latent 32x8x8; nearest upsample+conv 32-24-16-8-3',
        'local_fraction': .25, 'local_rounding': 'ceil',
        'global_aggregation': 'valid-pixel-weighted patch MSE',
        'quantile': .95, 'quantile_method': 'linear',
        'decision': 'score > threshold', 'threads': threads, 'device': 'cpu',
        'bootstrap_replicates': 2000, 'bootstrap_seed': 42,
        'versions': {
            'python': sys.version, 'torch': torch.__version__,
            'numpy': np.__version__, 'pillow': pillow_version,
            'matplotlib': matplotlib.__version__, 'sklearn': sklearn.__version__,
            'platform': platform.platform(),
        },
        'dataset_root_at_training': str(root.resolve()),
        'status': 'training',
    }


def validate_manifest(root, manifest):
    for name, expected in manifest['sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Изменён исходный файл: {name}')
    groups = [set(manifest[k]) for k in ['training', 'validation', 'calibration']]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Пересекаются группы манифеста')


def diagnostics_and_calibration(model, root, out, config, manifest, mean, history):
    # Диагностика на Validation, затем пороги на отдельной Calibration.
    model.eval()
    baseline_mse, input_std, output_std, mae = [], [], [], []
    validation_mse = []
    for i, name in enumerate(manifest['validation']):
        scores, x, y, mask, shape = predict(model, root / name)
        validation_mse.append(scores['global_mse'])
        errors = masked_errors(x, mean.expand_as(x), mask).numpy()
        weights = mask.sum((1, 2, 3)).numpy()
        baseline_mse.append(aggregate(errors, weights)['global_mse'])
        # Пространственная вариативность по каналам, без дополнения.
        valid = mask[:, 0].bool()
        xin = x.permute(1, 0, 2, 3)[:, valid]
        yout = y.permute(1, 0, 2, 3)[:, valid]
        input_std.append(float(xin.std(dim=1).mean()))
        output_std.append(float(yout.std(dim=1).mean()))
        mae.append(float((xin - yout).abs().mean()))
        if i < 3:
            reconstruction_from_patches(
                x, y, mask, shape, out / f'validation_reconstruction_{i}.png',
            )
    diag = {
        'best_validation_mse': float(np.mean(validation_mse)),
        'constant_baseline_validation_mse': float(np.mean(baseline_mse)),
        'input_spatial_std': float(np.mean(input_std)),
        'output_spatial_std': float(np.mean(output_std)),
        'validation_mae': float(np.mean(mae)),
        'training_loss_decreased': history[-1]['training_mse'] < history[0]['training_mse'],
    }
    diag['possible_constant_output'] = diag['output_spatial_std'] < .1 * diag['input_spatial_std']
    diag['better_than_constant_baseline'] = (
        diag['best_validation_mse'] < diag['constant_baseline_validation_mse']
    )
    save_json(out / 'diagnostics.json', diag)
    rows = []
    for name in manifest['calibration']:
        scores, *_ = predict(model, root / name)
        rows.append({'file': name, **scores})
    # Каждый метод получает свой порог только по нормальным Calibration.
    config['thresholds'] = {
        method: calibration_threshold([row[method] for row in rows])
        for method in ['global_mse', 'local_patch']
    }
    config['model_sha256'] = digest(out / 'best_model.pt')
    config['manifest_sha256'] = digest(out / 'split_manifest.json')
    config['status'] = 'trained_and_calibrated'
    save_json(out / 'calibration_scores.json', rows)
    save_json(out / 'config.json', config)
    (out / 'config.sha256').write_text(digest(out / 'config.json') + '\n', encoding='utf-8')
    write_training_report(out)
    print('Обучение и калибровка завершены.', flush=True)


def train(root, out, epochs=40, threads=2, resume=False):
    root, out = Path(root), Path(out)
    if not resume and out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} не пуст: выберите новую папку --output или --resume')
    if resume and not (out / 'last_checkpoint.pt').is_file():
        raise ValueError('Нет last_checkpoint.pt для точного продолжения')
    out.mkdir(parents=True, exist_ok=True)
    seed_all(threads=threads)
    if resume:
        config = json.loads((out / 'config.json').read_text())
        if config['status'] == 'trained_and_calibrated':
            raise ValueError('Эксперимент уже завершён; продолжение запрещено')
        if epochs != config['epochs_max'] or threads != config['threads']:
            raise ValueError('--epochs и --threads должны совпадать с исходным запуском')
        manifest = json.loads((out / 'split_manifest.json').read_text())
        validate_manifest(root, manifest)
    else:
        manifest = split(root)
        config = make_config(root, epochs, threads)
        save_json(out / 'split_manifest.json', manifest)
        save_json(out / 'config.json', config)

    dataset = TrainingPatches(root, manifest['training'])
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=0, generator=generator,
    )
    # Средний RGB по всем настоящим пикселям Training.
    rgb_sum = np.zeros(3, dtype=np.float64)
    pixels = 0
    for i in range(len(dataset.paths)):
        image = dataset.image(i)
        rgb_sum += image.sum(axis=(0, 1), dtype=np.float64)
        pixels += image.shape[0] * image.shape[1]
    mean = torch.tensor(rgb_sum / (pixels * 255), dtype=torch.float32).reshape(1, 3, 1, 1)
    save_json(out / 'baseline.json', {
        'training_rgb_mean': mean.flatten().tolist(),
        'definition': 'spatially constant RGB image',
    })
    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best, stale, history, start = float('inf'), 0, [], 1
    if resume:
        # Полный checkpoint содержит Adam и RNG; загружаются только свои файлы.
        checkpoint = torch.load(out / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
        if checkpoint['manifest_sha256'] != digest(out / 'split_manifest.json'):
            raise ValueError('Манифест не совпадает с checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        history = checkpoint['history']
        best, stale = checkpoint['best'], checkpoint['stale']
        config['best_epoch'] = checkpoint['best_epoch']
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        generator.set_state(checkpoint['loader_rng'])
        start = history[-1]['epoch'] + 1
    # При продолжении восстанавливаются и веса, и состояние оптимизатора и RNG.
    for epoch in range(start, epochs + 1):
        if stale >= config['patience']:
            break
        started = time.monotonic()
        model.train()
        total, count = 0., 0
        for x in loader:
            optimizer.zero_grad()
            loss = (x - model(x)).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('NaN/Inf в training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('NaN/Inf в градиентах')
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        # Validation не обновляет веса и использует фиксированную сетку фрагментов.
        model.eval()
        validation = []
        for name in manifest['validation']:
            scores, *_ = predict(model, root / name)
            validation.append(scores['global_mse'])
        value = float(np.mean(validation))
        history.append({'epoch': epoch, 'training_mse': total / count, 'validation_mse': value})
        # Выбор модели зависит только от Validation; stale считает эпохи без улучшения.
        if value < best:
            best, stale = value, 0
            atomic_torch_save(model.state_dict(), out / 'best_model.pt')
            config['best_epoch'] = epoch
        else:
            stale += 1
        # Сохраняем состояние последней эпохи для точного продолжения после сбоя.
        atomic_torch_save({
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'history': history, 'best': best, 'stale': stale,
            'best_epoch': config['best_epoch'],
            'manifest_sha256': digest(out / 'split_manifest.json'),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(), 'loader_rng': generator.get_state(),
        }, out / 'last_checkpoint.pt')
        save_json(out / 'config.json', config)
        save_json(out / 'history.json', history)
        history_plot(history, out / 'training.png')
        print(
            f'Эпоха {epoch:02d}: train={total/count:.6f}, validation={value:.6f}, '
            f'{time.monotonic()-started:.1f} с', flush=True,
        )
    # Диагностика и калибровка выполняются с лучшими, а не последними весами.
    model.load_state_dict(torch.load(out / 'best_model.pt', map_location='cpu', weights_only=True))
    diagnostics_and_calibration(model, root, out, config, manifest, mean, history)
