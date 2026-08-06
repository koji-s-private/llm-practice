"""Issue #72: プロダクト感のあるUI/UXへのリニューアルに関する静的検証テスト。

`.streamlit/config.toml`（Streamlitのカスタムテーマ）と、`app.py` 内の
`st.set_page_config(...)` 呼び出しの引数（page_title/page_icon/layout）は、
`streamlit.testing.v1.AppTest` の要素ツリーからは取得できない
（ページのメタ情報であり、描画される要素ではないため）。
そのため `test_ci_config.py` と同様にファイルをそのまま静的に読み込み・
パースして検証する。
"""

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
THEME_CONFIG_PATH = REPO_ROOT / ".streamlit" / "config.toml"
APP_PATH = REPO_ROOT / "app.py"

# 6桁の16進数カラーコード（#RRGGBB形式）かどうかを検証する簡易チェック用
_HEX_COLOR_KEYS = ("primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor")


def _is_hex_color(value: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("#")
        and len(value) == 7
        and all(c in "0123456789abcdefABCDEF" for c in value[1:])
    )


def _load_theme_config() -> dict:
    import tomllib

    with THEME_CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


def test_streamlit_config_toml_exists():
    """正常系: `.streamlit/config.toml` がリポジトリ直下に存在する
    （Streamlitはこの固定パスのみを読み込むため、配置場所自体が正しいことが前提条件）。"""
    assert THEME_CONFIG_PATH.is_file()


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllibはPython 3.11以降のみ標準搭載")
def test_streamlit_config_toml_is_valid_toml_with_theme_section():
    """正常系: 有効なTOMLとしてパースでき、`[theme]` セクションを持つ。"""
    config = _load_theme_config()
    assert "theme" in config


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllibはPython 3.11以降のみ標準搭載")
def test_streamlit_config_toml_defines_all_expected_theme_keys():
    """正常系: プロトタイプ感を脱するために必要な配色・フォントのキーが揃っている。"""
    theme = _load_theme_config()["theme"]
    expected_keys = {
        "base",
        "primaryColor",
        "backgroundColor",
        "secondaryBackgroundColor",
        "textColor",
        "font",
    }
    assert expected_keys <= theme.keys()


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllibはPython 3.11以降のみ標準搭載")
def test_streamlit_config_toml_color_values_are_valid_hex_codes():
    """境界値: 各配色キーの値が、Streamlitが受け付ける `#RRGGBB` 形式の6桁16進数である
    （例えば `#4F46E5` のような3桁省略形や `red` のような名前指定ではないことを確認する）。"""
    theme = _load_theme_config()["theme"]
    for key in _HEX_COLOR_KEYS:
        assert _is_hex_color(theme[key]), f"{key}={theme[key]!r} is not a valid #RRGGBB color"


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllibはPython 3.11以降のみ標準搭載")
def test_streamlit_config_toml_base_is_light():
    """異常系境界値: `base` に想定外の値（例: 誤字 "Light" や未対応の値）が
    紛れ込んでいないことを確認する。Streamlitが受け付けるのは "light" / "dark" のみ。"""
    theme = _load_theme_config()["theme"]
    assert theme["base"] in {"light", "dark"}
    assert theme["base"] == "light"


def _find_set_page_config_call(tree: ast.Module) -> ast.Call:
    """`app.py` のASTから `st.set_page_config(...)` の呼び出しノードを探す。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "set_page_config":
            return node
    raise AssertionError("st.set_page_config(...) の呼び出しが app.py 内に見つかりません")


def _load_set_page_config_kwargs() -> dict:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    call = _find_set_page_config_call(tree)
    kwargs = {}
    for kw in call.keywords:
        # 各引数は文字列/数値などのリテラルのみを想定しているため ast.literal_eval で十分
        kwargs[kw.arg] = ast.literal_eval(kw.value)
    return kwargs


def test_set_page_config_uses_doclore_branding():
    """正常系: `st.set_page_config` のタイトル・アイコンがリニューアル後の
    ブランド名「Doclore」に統一されている（プロトタイプ感のあった旧名称
    「llm-practice RAGチャット」/「📚」に戻っていないことも合わせて確認する）。"""
    kwargs = _load_set_page_config_kwargs()
    assert kwargs["page_title"] == "Doclore | ドキュメントAIアシスタント"
    assert kwargs["page_icon"] == "📖"
    assert kwargs["layout"] == "centered"
    assert "llm-practice" not in kwargs["page_title"]
