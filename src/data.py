# Разделение исходников и фрагменты без изменения исходного разрешения.
import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def files(folder):
    # Стабильный порядок файлов, независимо от порядка файловой системы.
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in EXT
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    # Запись через временный файл защищает JSON при прерывании.
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )
    temporary.replace(path)


def inspect(root):
    # Проверка файлов и метаданных; модель и оценки аномальности не используются.
    root = Path(root)
    if not (root / 'train/good').is_dir() or not (root / 'test/good').is_dir():
        raise ValueError(f'Не найдены train/good и test/good в {root}')
    result = {}
    for branch in ['train', 'test', 'ground_truth']:
        for folder in sorted((root / branch).glob('*')):
            if not folder.is_dir():
                continue
            counts = Counter()
            for p in files(folder):
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    counts[f'{im.format}, {im.width}*{im.height}, {im.mode}'] += 1
            result[str(folder.relative_to(root))] = {
                'count': sum(counts.values()), 'formats': dict(counts),
            }
    for folder in sorted((root / 'test').iterdir()):
        if folder.is_dir() and folder.name != 'good':
            for p in files(folder):
                mask = root / 'ground_truth' / folder.name / (p.stem + '_mask.png')
                if not mask.is_file():
                    raise ValueError(f'Отсутствует маска: {p}')
    return result


def split(root, seed=42):
    # Делим исходные файлы: все фрагменты одного изображения остаются в его группе.
    paths = files(Path(root) / 'train/good')
    if len(paths) < 20:
        raise ValueError('Недостаточно train/good для трёх независимых групп')
    order = np.random.default_rng(seed).permutation(len(paths))
    a, b = int(.7 * len(paths)), int(.15 * len(paths))
    groups = {
        'training': order[:a],
        'validation': order[a:a+b],
        'calibration': order[a+b:],
    }
    manifest = {
        key: [str(paths[i].relative_to(root)) for i in ids]
        for key, ids in groups.items()
    }
    hashes = {str(p.relative_to(root)): digest(p) for p in paths}
    if len(set(hashes.values())) != len(hashes):
        raise ValueError('Найдены дубликаты содержимого train/good; требуется групповое разбиение')
    manifest['sha256'] = hashes
    manifest['seed'] = seed
    return manifest


def load_image(path, patch=128):
    with Image.open(path) as im:
        x = np.asarray(im.convert('RGB'), dtype=np.float32) / 255
    h, w = x.shape[:2]
    ph, pw = (-h) % patch, (-w) % patch
    # Единицы отмечают исходные пиксели, нули — добавленные края.
    mask = np.pad(
        np.ones((h, w, 1), dtype=np.float32), ((0, ph), (0, pw), (0, 0)),
    )
    x = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    return (
        torch.from_numpy(x.copy()).permute(2, 0, 1),
        torch.from_numpy(mask).permute(2, 0, 1),
    )


def grid(path):
    # Порядок фрагментов: слева направо, сверху вниз; весь оригинал покрыт.
    x, mask = load_image(path)

    def cut(tensor):
        return (
            tensor.unfold(1, 128, 128).unfold(2, 128, 128)
            .permute(1, 2, 0, 3, 4).reshape(-1, tensor.shape[0], 128, 128)
        )

    return cut(x), cut(mask), tuple(x.shape[-2:])


def assemble_patches(patches, shape):
    # Сборка сетки слева направо, сверху вниз; работает и для маски.
    height, width = shape
    channels = patches.shape[1]
    return (
        patches.reshape(height//128, width//128, channels, 128, 128)
        .permute(2, 0, 3, 1, 4).reshape(channels, height, width)
    )


class TrainingPatches(Dataset):
    def __init__(self, root, names, patches_per_image=8):
        self.paths = [Path(root) / name for name in names]
        self.repeats = patches_per_image
        # Кэш исходников ограничен, чтобы не заполнять память на CPU.
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = 512 * 1024**2

    def image(self, index):
        path = self.paths[index]
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        with Image.open(path) as im:
            image = np.array(im.convert('RGB'))
        while self.cache and self.cache_bytes + image.nbytes > self.cache_limit:
            _, removed = self.cache.popitem(last=False)
            self.cache_bytes -= removed.nbytes
        if image.nbytes <= self.cache_limit:
            self.cache[path] = image
            self.cache_bytes += image.nbytes
        return image

    def __len__(self):
        return len(self.paths) * self.repeats

    def __getitem__(self, index):
        image = self.image(index // self.repeats)
        h, w = image.shape[:2]
        if min(h, w) < 128:
            raise ValueError('Исходное изображение меньше 128 пикселей')
        # Обучающие фрагменты целиком лежат в исходном изображении.
        y = torch.randint(h - 127, ()).item()
        z = torch.randint(w - 127, ()).item()
        crop = image[y:y+128, z:z+128].astype(np.float32) / 255
        x = torch.from_numpy(crop).permute(2, 0, 1)
        # Аугментации применяются только к случайным тренировочным фрагментам.
        x = torch.rot90(x, torch.randint(4, ()).item(), (1, 2))
        if torch.rand(()) < .5:
            x = x.flip(2)
        return x.contiguous()


def masked_errors(x, reconstructed, mask):
    # MSE каждого фрагмента, усреднённая по RGB и настоящим пикселям.
    # В знаменателе учитываем только настоящие пиксели и три RGB-канала.
    errors = (
        ((x - reconstructed).square() * mask).sum((1, 2, 3))
        / (mask.sum((1, 2, 3)) * 3)
    )
    if not torch.isfinite(errors).all():
        raise ValueError('Обнаружены NaN/Inf в ошибках восстановления')
    return errors


def aggregate(errors, weights):
    # Global — средняя по исходным пикселям; Local — ceil(25%) худших фрагментов.
    k = max(1, int(np.ceil(len(errors) * .25)))
    return {
        'global_mse': float(np.average(errors, weights=weights)),
        'local_patch': float(np.sort(errors)[-k:].mean()),
    }
