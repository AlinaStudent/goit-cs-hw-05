#!/usr/bin/env python3
# Загрузка текста по URL, подсчет частоты слов в парадигме MapReduce
# (многопоточно) и визуализация топ-слов.

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import requests


# ------------------------ Загрузка и препроцессинг ------------------------- #

def get_text(url: str, *, timeout: int = 20) -> Optional[str]:
    # Загружает текст по URL, возвращает содержимое либо None при ошибке
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.encoding or "utf-8"
        return resp.text
    except requests.RequestException as exc:
        logging.error("Не удалось загрузить URL %s: %s", url, exc)
        return None


def tokenize(text: str) -> List[str]:
    # Превращает текст в список слов в нижнем регистре без пунктуации
    text = text.lower()
    tokens = re.split(r"[^0-9a-zA-Zа-яёА-ЯЁ]+", text)
    return [t for t in tokens if t]


# ------------------------------- MapReduce -------------------------------- #

def map_function(word: str) -> Tuple[str, int]:
    return word, 1


def shuffle_function(mapped_values):
    buckets: Dict[str, List[int]] = defaultdict(list)
    for key, value in mapped_values:
        buckets[key].append(value)
    return buckets.items()


def reduce_function(key_values: Tuple[str, List[int]]) -> Tuple[str, int]:
    key, values = key_values
    return key, sum(values)


def map_reduce(words: List[str],
               *,
               threads: int | None = None,
               restrict_to: Optional[set[str]] = None) -> Dict[str, int]:
    # Выполняет MapReduce: map, shuffle, reduce в потоках
    if restrict_to:
        words = [w for w in words if w in restrict_to]

    with ThreadPoolExecutor(max_workers=threads) as pool:
        mapped = list(pool.map(map_function, words))

    shuffled = list(shuffle_function(mapped))

    with ThreadPoolExecutor(max_workers=threads) as pool:
        reduced = dict(pool.map(reduce_function, shuffled))

    return reduced


# ------------------------------- Визуализация ------------------------------ #

def visualize_top_words(freqs: Dict[str, int],
                        top_n: int = 10,
                        *,
                        title: str = "Топ слов по частоте") -> None:
    # Строит столбиковую диаграмму топ-N слов по убыванию частоты
    if not freqs:
        logging.warning("Нет данных для визуализации.")
        return

    top_items = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    words, counts = zip(*top_items)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(words)), counts)
    plt.xticks(range(len(words)), words, rotation=45, ha="right")
    plt.ylabel("Частота")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -------------------------------- CLI/MAIN -------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Загружает текст по URL, считает частоты слов (MapReduce), "
                    "визуализирует топ-слова."
    )
    parser.add_argument("url", help="URL с текстом (plain text).")
    parser.add_argument("-n", "--top", type=int, default=10,
                        help="Сколько топ-слов показать (по умолчанию 10).")
    parser.add_argument("-t", "--threads", type=int, default=None,
                        help="Число потоков для Map и Reduce (по умолчанию — auto).")
    parser.add_argument("--only",
                        help="Путь к файлу со списком слов (по одному в строке) — "
                             "считать частоты только этих слов.")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Подробность логов (-v, -vv).")
    return parser.parse_args()


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else (
        logging.INFO if verbosity == 1 else logging.DEBUG
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_wordlist(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return {line.strip().lower() for line in fh if line.strip()}
    except OSError as exc:
        logging.error("Не удалось прочитать список слов %s: %s", path, exc)
        return None


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    text = get_text(args.url)
    if text is None:
        print("Ошибка: не удалось получить текст по URL.")
        return

    words = tokenize(text)
    logging.info("Токенов: %d", len(words))

    restrict = load_wordlist(args.only)
    freqs = map_reduce(words, threads=args.threads, restrict_to=restrict)
    logging.info("Уникальных слов: %d", len(freqs))

    visualize_top_words(freqs, top_n=args.top,
                        title=f"Топ {args.top} слов — {args.url}")


if __name__ == "__main__":
    main()
