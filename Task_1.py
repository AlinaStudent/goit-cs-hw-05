
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
import shutil
from typing import List

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-s", "--source", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("-c", "--concurrency", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--follow-symlinks", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args()

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

async def read_folder(
    src: Path,
    dst: Path,
    *,
    concurrency: int,
    follow_symlinks: bool = False,
) -> None:
    if not await _exists(src):
        logging.error("Исходная папка не существует: %s", src)
        return
    if not await _is_dir(src):
        logging.error("Исходный путь не является папкой: %s", src)
        return

    await _mkdir(dst, parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    tasks: List[asyncio.Task[None]] = []

    for root, _, files in await asyncio.to_thread(lambda: list(os.walk(src, followlinks=follow_symlinks))):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            if not file_path.exists():
                logging.warning("Пропуск: %s", file_path)
                continue
            tasks.append(asyncio.create_task(_guarded_copy(file_path, dst, sem)))

    if not tasks:
        logging.info("Файлы не найдены: %s", src)
        return

    logging.info("Файлов: %d; параллелизм=%d", len(tasks), concurrency)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            logging.exception("Ошибка", exc_info=r)
    logging.info("Готово. Обработано: %d", len(tasks))

async def _guarded_copy(file_path: Path, dst_root: Path, sem: asyncio.Semaphore) -> None:
    try:
        async with sem:
            await copy_file(file_path, dst_root)
    except Exception as exc:  # noqa: BLE001
        logging.exception("Не удалось: '%s': %s", file_path, exc)

async def copy_file(src_file: Path, dst_root: Path) -> None:
    if not await _is_file(src_file):
        logging.debug("Пропуск: %s", src_file)
        return
    ext = src_file.suffix.lower().lstrip(".") or "no_extension"
    target_dir = dst_root / ext
    await _mkdir(target_dir, parents=True, exist_ok=True)
    target_path = await _unique_path(target_dir / src_file.name)
    await asyncio.to_thread(shutil.copy2, src_file, target_path)
    logging.debug("%s -> %s", src_file, target_path)

async def _exists(path: Path) -> bool:
    return await asyncio.to_thread(path.exists)

async def _is_dir(path: Path) -> bool:
    return await asyncio.to_thread(path.is_dir)

async def _is_file(path: Path) -> bool:
    return await asyncio.to_thread(path.is_file)

async def _mkdir(path: Path, *, parents: bool, exist_ok: bool) -> None:
    await asyncio.to_thread(path.mkdir, parents=parents, exist_ok=exist_ok)

async def _unique_path(path: Path) -> Path:
    if not await _exists(path):
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    n = 1
    while True:
        candidate = parent / f"{stem} ({n}){suffix}"
        if not await _exists(candidate):
            return candidate
        n += 1

async def _amain(args: argparse.Namespace) -> None:
    try:
        await read_folder(
            src=args.source,
            dst=args.output,
            concurrency=max(1, int(args.concurrency)),
            follow_symlinks=bool(args.follow_symlinks),
        )
    except Exception as exc:  # noqa: BLE001
        logging.exception("Фатальная ошибка: %s", exc)

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    asyncio.run(_amain(args))

if __name__ == "__main__":
    main()
