# Графики и восстановления в PNG без графического интерфейса.
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from .data import assemble_patches

plt.rcParams['font.family'] = 'DejaVu Sans'
METHOD_NAMES = {'global_mse': 'A: Global MSE', 'local_patch': 'B: Local Patch Error'}


def finish(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def history_plot(history, path):
    plt.figure()
    epochs = [row['epoch'] for row in history]
    plt.plot(epochs, [r['training_mse'] for r in history], label='Обучение: случайные фрагменты')
    plt.plot(epochs, [r['validation_mse'] for r in history], label='Валидация: фиксированная сетка')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.title('Ошибка восстановления нормальной текстуры')
    plt.legend()
    finish(path)


def reconstruction(x, y, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    views = [
        x.permute(1, 2, 0).numpy(), y.permute(1, 2, 0).numpy(),
        (x-y).square().mean(0).numpy(),
    ]
    titles = ['Исходное изображение', 'Восстановление', 'Карта квадратичной ошибки']
    for ax, view, title in zip(axes, views, titles):
        im = ax.imshow(view, cmap='magma' if view.ndim == 2 else None)
        ax.set_title(title)
        ax.axis('off')
        if view.ndim == 2:
            fig.colorbar(im, ax=ax, fraction=.046)
    finish(path)


def reconstruction_from_patches(x, y, mask, shape, path):
    # Дополненные пиксели не показываются на иллюстрациях.
    image = assemble_patches(x, shape)
    restored = assemble_patches(y, shape)
    valid = assemble_patches(mask, shape)
    real_h = int(valid[0, :, 0].sum())
    real_w = int(valid[0, 0, :].sum())
    reconstruction(image[:, :real_h, :real_w], restored[:, :real_h, :real_w], path)


def evaluation_plots(rows, metrics, thresholds, out):
    out = Path(out)
    y = np.array([row['label'] for row in rows])
    # ROC использует непрерывные оценки ошибок, а не решения по порогам.
    plt.figure()
    for method in thresholds:
        scores = np.array([row[method] for row in rows])
        fpr, tpr, _ = roc_curve(y, scores)
        plt.plot(fpr, tpr, label=f'{METHOD_NAMES[method]}: AUC={metrics[method]["roc_auc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('ROC-кривые двух заранее заданных методов')
    plt.legend()
    finish(out / 'roc.png')
    for method, threshold in thresholds.items():
        scores = np.array([row[method] for row in rows])
        plt.figure()
        # Одинаковые границы столбцов для обоих классов.
        bins = np.histogram_bin_edges(scores, bins=15)
        for label, name in [(0, 'Нормальные'), (1, 'Дефектные')]:
            plt.hist(scores[y == label], bins=bins, alpha=.5, label=name)
        plt.axvline(threshold, color='black', label='Калибровочный порог')
        plt.xlabel('Оценка аномальности')
        plt.ylabel('Изображений')
        plt.title(METHOD_NAMES[method])
        plt.legend()
        finish(out / f'distribution_{method}.png')
        matrix = np.array(metrics[method]['confusion_matrix'])
        plt.figure()
        plt.imshow(matrix, cmap='Blues')
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(matrix[i, j]), ha='center', va='center')
        plt.xticks([0, 1], ['Норма', 'Дефект'])
        plt.yticks([0, 1], ['Норма', 'Дефект'])
        plt.xlabel('Предсказание')
        plt.ylabel('Истина')
        plt.title(METHOD_NAMES[method])
        finish(out / f'confusion_{method}.png')
    names = ['roc_auc', 'precision', 'recall', 'f1', 'false_positive_rate']
    positions = np.arange(len(names))
    plt.figure(figsize=(10, 4))
    for i, method in enumerate(thresholds):
        plt.bar(
            positions + (i-.5)*.35, [metrics[method][name] for name in names],
            width=.35, label=METHOD_NAMES[method],
        )
    plt.xticks(positions, ['ROC-AUC', 'Precision', 'Recall', 'F1', 'FPR'])
    plt.ylim(0, 1.05)
    plt.title('Описательное сравнение без выбора метода по тесту')
    plt.legend()
    finish(out / 'comparison.png')
