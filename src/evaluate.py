# Окончательная оценка с зафиксированными моделью и порогами.
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from .data import digest, files, save_json
from .model import Autoencoder
from .reporting import write_evaluation_report
from .train import predict, seed_all
from .visualization import evaluation_plots, reconstruction_from_patches


def metrics(y, scores, threshold, replicates=2000, seed=42):
    # Метрики изображений и стратифицированный bootstrap-интервал ROC-AUC.
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape:
        raise ValueError('Метки и оценки должны быть одномерными массивами одной длины')
    if set(y.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Для метрик требуются оба класса и конечные оценки')
    if replicates < 1 or not np.isfinite(threshold):
        raise ValueError('Некорректный порог или число бутстреп-выборок')
    # Равенство порогу означает норму; строки матрицы — истинные классы.
    predicted = scores > threshold
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in [0, 1]]
    aucs = []
    # Перевыбираем целые изображения внутри каждого класса, а не их фрагменты.
    for _ in range(replicates):
        indices = np.concatenate([
            rng.choice(group, len(group), replace=True) for group in groups
        ])
        aucs.append(roc_auc_score(y[indices], scores[indices]))
    return {
        'roc_auc': float(roc_auc_score(y, scores)),
        'roc_auc_ci95': np.quantile(aucs, [.025, .975]).tolist(),
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'false_positive_rate': float(fp / (tn + fp)),
        'confusion_matrix': matrix.tolist(), 'threshold': threshold,
    }


def validate_protocol(config):
    # Проверка совместимости сохранённого протокола с реализацией.
    expected = {
        'patch_size': 128, 'local_fraction': .25, 'local_rounding': 'ceil',
        'global_aggregation': 'valid-pixel-weighted patch MSE',
        'preprocessing': 'native RGB; reflect pad bottom/right to multiple of 128; mask padding',
        'decision': 'score > threshold',
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Конфигурация {key} не соответствует реализации')
    if set(config.get('thresholds', {})) != {'global_mse', 'local_patch'}:
        raise ValueError('Отсутствуют пороги обоих методов')
    if not np.isfinite(list(config['thresholds'].values())).all():
        raise ValueError('Пороги содержат NaN/Inf')


def evaluate(root, out, confirmed=False):
    if not confirmed:
        raise ValueError(
            'Официальный тест требует --confirm-final-evaluation; '
            'запуск раскрывает тестовые результаты',
        )
    root, out = Path(root), Path(out)
    config = json.loads((out / 'config.json').read_text())
    if config.get('status') != 'trained_and_calibrated':
        raise ValueError('Обучение и калибровка не завершены')
    if digest(out / 'config.json') != (out / 'config.sha256').read_text().strip():
        raise ValueError('Изменены зафиксированные настройки или пороги')
    # Проверки не позволяют незаметно оценить изменённые настройки или веса.
    validate_protocol(config)
    for name, key in [
        ('best_model.pt', 'model_sha256'),
        ('split_manifest.json', 'manifest_sha256'),
    ]:
        if digest(out / name) != config[key]:
            raise ValueError(f'Изменён зафиксированный файл {name}')
    target = out / 'final_evaluation'
    if target.exists():
        raise ValueError('Окончательный тест уже запускался: папка final_evaluation существует')
    manifest = json.loads((out / 'split_manifest.json').read_text())
    training_hashes = set(manifest['sha256'].values())
    for name, expected in manifest['sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Изменён исходный файл {name}')
    test = [
        path for folder in sorted((root / 'test').iterdir()) if folder.is_dir()
        for path in files(folder)
    ]
    if not test or not any(p.parent.name == 'good' for p in test) or not any(p.parent.name != 'good' for p in test):
        raise ValueError('Неполный официальный тест')
    test_hashes = {str(p.relative_to(root)): digest(p) for p in test}
    if training_hashes.intersection(test_hashes.values()):
        raise ValueError('Дубликаты между train и test')

    # Даже прерванная оценка раскрывает тест: её каталог не перезаписывается.
    target.mkdir()
    save_json(target / 'frozen_config.json', config)
    save_json(target / 'test_manifest.json', test_hashes)
    seed_all(threads=config['threads'])
    model = Autoencoder()
    model.load_state_dict(torch.load(out / 'best_model.pt', map_location='cpu', weights_only=True))
    model.eval()
    rows, seen = [], set()
    for path in test:
        scores, x, y, mask, shape = predict(model, path, config['batch_size'])
        kind = path.parent.name
        rows.append({
            'file': str(path.relative_to(root)), 'defect_type': kind,
            'label': int(kind != 'good'), **scores,
        })
        if kind not in seen:
            reconstruction_from_patches(x, y, mask, shape, target / f'example_{kind}.png')
            seen.add(kind)
    save_json(target / 'scores.json', rows)
    result, by_type = {}, {}
    for method, threshold in config['thresholds'].items():
        result[method] = metrics(
            [row['label'] for row in rows], [row[method] for row in rows], threshold,
            config['bootstrap_replicates'], config['bootstrap_seed'],
        )
        # Разбивка по типам описывает результат теста и не используется для настройки.
        by_type[method] = {
            kind: {
                'n': sum(row['defect_type'] == kind for row in rows),
                'flagged_fraction': float(np.mean([
                    row[method] > threshold for row in rows if row['defect_type'] == kind
                ])),
            }
            for kind in sorted(seen)
        }
    save_json(target / 'metrics.json', result)
    save_json(target / 'by_defect_type.json', by_type)
    evaluation_plots(rows, result, config['thresholds'], target)
    write_evaluation_report(rows, result, config, target / 'report.md')
    print(json.dumps(result, ensure_ascii=False, indent=2))
