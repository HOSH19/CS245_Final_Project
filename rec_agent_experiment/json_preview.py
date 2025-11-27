#!/usr/bin/env python3
"""Quick preview tool for the large JSONL files in ``data/``.

支持两种用法：
1. 在 IDE 中直接运行，修改文件底部的 `run()` 参数即可。
2. 命令行调用（保留原 CLI 参数，方便脚本化）。

所有数据文件是 JSON Lines（每行一个 JSON 对象），脚本流式读取，不会占用太多内存。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional


DEFAULT_FILES: Mapping[str, str] = {
    "item": "data/item.json",
    "user": "data/user.json",
    "review": "data/review.json",
}


def iter_json_lines(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            yield json.loads(raw)


def preview(path: Path, limit: int, source: Optional[str]) -> Iterable[str]:
    count = 0
    needle = source.lower() if source else None
    for obj in iter_json_lines(path):
        if needle:
            # source 字段可能缺失或大小写不一致
            if str(obj.get("source", "")).lower() != needle:
                continue
        count += 1
        if count > limit:
            break

        obj_print = dict(obj)
        if "friends" in obj_print:
            raw_friends = obj_print.get("friends", [])
            if isinstance(raw_friends, str):
                friends_list = [
                    item.strip() for item in raw_friends.split(",") if item.strip()
                ]
            elif isinstance(raw_friends, list):
                friends_list = raw_friends
            else:
                friends_list = []
            obj_print["friend_count"] = len(friends_list)
            obj_print["friends"] = friends_list[:2]

        yield f"--- #{count} ---\n" + json.dumps(
            obj_print, ensure_ascii=False, indent=2
        )


def resolve_path(dataset: Optional[str], path: Optional[str], root: str) -> Path:
    if dataset:
        rel = DEFAULT_FILES[dataset]
        return Path(root).joinpath(rel).resolve()
    assert path is not None  # safeguarded by mutually exclusive group
    return Path(path).expanduser().resolve()


def run(
    *,
    dataset: Optional[str] = "item",
    path: Optional[str] = None,
    limit: int = 2,
    root: str = ".",
    source: Optional[str] = None,
) -> None:
    """Preview helper suitable for IDE 调试.

    Args:
        dataset: ``item`` / ``user`` / ``review``（与 data/ 下文件对应）。传 ``None`` 时使用 ``path``。
        path:    自定义文件路径（JSONL），优先级高于 ``dataset``。
        limit:   输出条数。
        root:    解析 dataset 的基准目录（默认当前目录）。
    """

    target = resolve_path(dataset if path is None else None, path, root)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")

    for chunk in preview(target, limit=max(0, limit), source=source):
        print(chunk)


if __name__ == "__main__":
    # IDE 运行时直接改这里的参数即可，命令行仍可传参覆盖
    run(dataset="item", limit=1, root=".", source="yelp")
    run(dataset="user", limit=1, root=".", source="yelp")
    run(dataset="review", limit=1, root=".", source="yelp")
