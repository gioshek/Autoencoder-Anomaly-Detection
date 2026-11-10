# Команды инспекции данных, обучения и окончательной оценки.
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Надёжность обнаружения дефектов: MVTec AD tile',
    )
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['inspect', 'train', 'evaluate']:
        command = sub.add_parser(name)
        command.add_argument(
            '--data-dir', type=Path,
            default=Path('data/mvtec_anomaly_detection/tile'),
            help='Путь к категории tile',
        )
        command.add_argument('--output', type=Path, default=Path('outputs/run_42'))
        if name == 'train':
            command.add_argument('--epochs', type=int, default=40)
            command.add_argument('--threads', type=int, default=2)
            command.add_argument(
                '--resume', action='store_true',
                help='Продолжить из last_checkpoint.pt',
            )
        if name == 'evaluate':
            command.add_argument('--confirm-final-evaluation', action='store_true')
    # Общие пути задаются для каждой команды; обучение и оценка запускаются отдельно.
    args = parser.parse_args()
    try:
        if args.command == 'inspect':
            from src.data import inspect, save_json
            result = inspect(args.data_dir)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Инспекция не занимает каталог будущего обучения.
            path = args.output.parent / (args.output.name + '_inspection.json')
            save_json(path, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == 'train':
            if not 1 <= args.epochs <= 70 or args.threads < 1:
                raise ValueError('epochs: 1–70; threads: >=1')
            from src.train import train
            train(args.data_dir, args.output, args.epochs, args.threads, args.resume)
        else:
            from src.evaluate import evaluate
            evaluate(args.data_dir, args.output, args.confirm_final_evaluation)
    except (ValueError, FileNotFoundError, NotADirectoryError, KeyError, RuntimeError, OSError) as exc:
        parser.exit(2, f'Ошибка: {exc}\n')


if __name__ == '__main__':
    main()
