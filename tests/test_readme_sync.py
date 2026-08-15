"""tests/README.md の一覧表と tests/ 配下の実ファイルの整合性を検証するテスト。

`tests/README.md` の「ファイル一覧と対応する実装ファイル」表は手動更新のため、
新しいテストファイルを追加した際に表の更新を忘れると陳腐化する。ここでは
`tests/` 配下に実在する `test_*.py` のファイル名集合と、README表中に記載
された `test_*.py` 形式のファイル名集合を突き合わせ、過不足がないことを検証する。
"""

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
README_PATH = TESTS_DIR / "README.md"

# 表の行のみを対象にする（「新しいテストを追加する際の命名規則」セクションの
# 説明文中の例示（`test_foo.py`）を誤って拾わないため）。
TABLE_ROW_PATTERN = re.compile(r"^\|\s*`(test_[A-Za-z0-9_]+\.py)`", re.MULTILINE)


def _actual_test_files() -> set[str]:
    return {path.name for path in TESTS_DIR.glob("test_*.py")}


def _readme_referenced_test_files() -> set[str]:
    readme_text = README_PATH.read_text(encoding="utf-8")
    return set(TABLE_ROW_PATTERN.findall(readme_text))


def test_readme_exists():
    assert README_PATH.exists()


def test_all_actual_test_files_are_listed_in_readme():
    missing = _actual_test_files() - _readme_referenced_test_files()
    assert not missing, f"tests/README.mdの表に記載が無いテストファイル: {sorted(missing)}"


def test_readme_does_not_reference_nonexistent_test_files():
    stale = _readme_referenced_test_files() - _actual_test_files()
    assert not stale, f"tests/README.mdの表に記載されているが実在しないテストファイル: {sorted(stale)}"
