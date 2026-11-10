# Материалы завершённого эксперимента

Основные файлы скопированы из `outputs/run_42_final/` без изменения исходных результатов:

- `training.png` — история обучения за 70 эпох.
- `roc.png`, `comparison.png`, `metrics.json` — графики и метрики из `final_evaluation/`.
- `example_good.png`, `example_crack.png`, `example_glue_strip.png`, `example_oil.png` — готовые иллюстрации из той же оценки: оригинал, восстановление и карта ошибки.
- `split_manifest.json` — исходное разбиение Training / Validation / Calibration, seed=42.

Метрики сохранены с полной точностью. `recall_by_defect.png` — дополнительная описательная диаграмма по пяти типам дефектов. Она построена только по сохранённым `final_evaluation/scores.json`, с исходными порогами из `metrics.json`; количества и Recall сверены с `by_defect_type.json`. Модель для диаграммы не запускалась.

Иллюстрации восстановлений скопированы из прежнего отчёта. По правилу сохранения примеров в оценке это первые файлы каждого типа в сортировке: `test/good/000.png` признан нормальным обоими методами, `test/crack/000.png` обнаружен обоими, `test/glue_strip/000.png` пропущен обоими. Решения проверены по сохранённым оценкам и порогам. Масляное пятно оставлено дополнительным примером.

## Источник и лицензия изображений

Источник исходных фотографий: **MVTec AD, категория tile**. Copyright 2019 MVTec Software GmbH.

- [Официальная страница датасета и условия использования](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
- [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- [Полный текст лицензии из исходного датасета](LICENSE-MVTec-AD.txt).

Файлы `example_*.png` содержат фотографии MVTec AD с добавленными восстановленными изображениями и картами ошибок. Эти составные иллюстрации распространяются по **CC BY-NC-SA 4.0**: необходимо указание источника, использование допускается только в некоммерческих целях, производные материалы распространяются на тех же условиях. Лицензия изображений не устанавливает лицензию программного кода проекта.

Библиографический источник: Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR, 2019.
