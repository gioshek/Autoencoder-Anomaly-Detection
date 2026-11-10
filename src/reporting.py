# Краткие отчёты по сохранённым результатам обучения и оценки.
import json
from pathlib import Path


def write_training_report(out):
    # Отчёт читает готовые файлы; модель и пороги здесь не пересчитываются.
    out = Path(out)
    config = json.loads((out / 'config.json').read_text())
    diagnostics = json.loads((out / 'diagnostics.json').read_text())
    manifest = json.loads((out / 'split_manifest.json').read_text())
    history = json.loads((out / 'history.json').read_text())
    if config['status'] != 'trained_and_calibrated':
        raise ValueError('Отчёт требует завершённого обучения и калибровки')
    lines = [
        '# Обучение и калибровка', '',
        f'Training / Validation / Calibration: {len(manifest["training"])} / '
        f'{len(manifest["validation"])} / {len(manifest["calibration"])}; seed={config["seed"]}.',
        f'Завершено эпох: {len(history)}; лучшая: {config["best_epoch"]}.', '',
        '| Диагностика на Validation | Значение |', '|---|---:|',
    ]
    for label, key in [
        ('MSE автоэнкодера', 'best_validation_mse'),
        ('MSE среднего RGB', 'constant_baseline_validation_mse'),
        ('MAE автоэнкодера', 'validation_mae'),
        ('Spatial std входа', 'input_spatial_std'),
        ('Spatial std восстановления', 'output_spatial_std'),
    ]:
        lines.append(f'| {label} | {diagnostics[key]:.8f} |')
    lines += ['', '| Метод | Порог Calibration |', '|---|---:|']
    for method, threshold in config['thresholds'].items():
        lines.append(f'| {method} | {threshold:.10f} |')
    lines += [
        '', 'Порог — 95-й процентиль Calibration с линейной интерполяцией.',
        'Решение: score > threshold. Метрики обнаружения здесь не вычисляются.', '',
        '[График обучения](training.png)',
    ]
    (out / 'TRAINING_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_evaluation_report(rows, result, config, path):
    normals = sum(row['label'] == 0 for row in rows)
    lines = [
        '# Окончательная оценка MVTec AD tile', '',
        f'Нормальных изображений: {normals}; дефектных: {len(rows)-normals}.',
        'Положительный класс — дефект. Пороги получены до теста на Calibration.', '',
        '| Метод | ROC-AUC (95% CI) | Precision | Recall | F1 | FPR | Порог |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for method, values in result.items():
        lo, hi = values['roc_auc_ci95']
        lines.append(
            f'| {method} | {values["roc_auc"]:.4f} ({lo:.4f}–{hi:.4f}) '
            f'| {values["precision"]:.4f} | {values["recall"]:.4f} '
            f'| {values["f1"]:.4f} | {values["false_positive_rate"]:.4f} '
            f'| {values["threshold"]:.8f} |'
        )
    lines += [
        '', f'Бутстреп: {config["bootstrap_replicates"]} повторений, стратификация по классу, '
        f'seed={config["bootstrap_seed"]}. Единица ресемплинга — изображение.',
        'CI не учитывает неопределённость обучения или калибровки порогов.', '',
        'Матрицы ошибок имеют вид [[TN, FP], [FN, TP]]:', '',
    ]
    lines += [f'- {method}: `{values["confusion_matrix"]}`' for method, values in result.items()]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
