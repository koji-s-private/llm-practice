"""app.py のエラーハンドリング・自動再同期のテスト。

`streamlit.testing.v1.AppTest` を使い、実際にStreamlitのスクリプト実行エンジン上で
app.py を動かして検証する。重い外部依存（埋め込みモデル・Chroma・LLM）は使わず、
app.py が直接 import している以下のシンボルを monkeypatch で軽量なフェイクに
差し替える:

- `ingest.sync_data_dir`（サイドバーの再同期ボタン・ファイルアップロード時・
  トップレベルの軽量シグネチャチェックがdata/の変化を検知した場合、の各所から呼ばれる）
- `ingest.data_dir_signature`（トップレベルで毎回呼ばれる軽量な
  変更検知。デフォルトでは実ファイルシステムを見るため、シグネチャの変化を
  意図的に起こしたいテストではmonkeypatchで差し替える）
- `rag_chain.build_agent`（フェイクエージェントを返す。`.invoke()` の成功/失敗を
  テストごとに切り替える）
- `memory.new_thread_id` / `memory.conversation_count` / `memory.save_conversation`

app.py はモジュールトップレベルで `from ingest import ... sync_data_dir` のように
シンボルをインポートしているため、`AppTest.run()` がスクリプトを実行する
「前」に対象モジュールの属性を monkeypatch しておく必要がある
（実行時に束縛される値がその時点の属性値になるため）。

同期トリガーの設計が変わった点に注意:
- 従来: チャット応答→会話ログ保存の直後に無条件で `sync_data_dir(verbose=False)` を
  同じturn内で呼んでいた。
- 現在: スクリプトのトップレベルで毎回 `data_dir_signature()`（ファイル数+最新mtimeの
  軽量比較。内容の読み込みは行わない）を計算し、前回値と異なる場合にのみ
  `sync_data_dir()` を呼ぶ。チャット応答後の会話ログ保存自体はその場では同期をトリガー
  せず、次にスクリプトが再実行されたタイミング（次のチャット送信・ボタン操作等）で
  シグネチャの変化が検知されて初めて同期される。
"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

import ingest
import memory
import rag_chain

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class _FakeAgent:
    """rag_chain.build_agent() の代わりに使うフェイクエージェント。"""

    def __init__(self, answer="テスト回答です", exc=None):
        self.answer = answer
        self.exc = exc
        self.invoke_calls = []

    def invoke(self, payload):
        self.invoke_calls.append(payload)
        if self.exc is not None:
            raise self.exc
        return {"messages": payload["messages"] + [AIMessage(content=self.answer)]}


def _ok_sync(verbose=False):
    return {"added": [], "updated": [], "removed": [], "failed": []}


@pytest.fixture(autouse=True)
def _patch_light_dependencies(monkeypatch):
    """全テスト共通: 起動時同期とmemory系を軽量フェイクに差し替える。

    build_agent・data_dir_signatureの挙動・チャット後の同期挙動は、
    各テストが個別に上書きする。
    """
    monkeypatch.setattr(ingest, "sync_data_dir", _ok_sync)
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: _FakeAgent())
    monkeypatch.setattr(memory, "new_thread_id", lambda: "thread-test")
    monkeypatch.setattr(memory, "conversation_count", lambda thread_id: 0)
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: None)


def _run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run()
    return at


# --- 1. 起動時の sync_data_dir() 呼び出し（_sync_and_report経由） ---


def test_startup_sync_success_shows_no_error():
    """正常系: 起動時同期が成功すればエラーは表示されず、エージェントも構築される。"""
    at = _run_app()

    assert at.exception == []
    assert at.error == []
    assert "agent" in at.session_state


def test_startup_sync_failure_shows_error_but_app_keeps_running(monkeypatch):
    """異常系: 起動時同期が例外を送出しても、st.error表示のみでクラッシュせず、
    エージェント構築（後続処理）は継続される。"""

    def failing_sync(verbose=False):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ingest, "sync_data_dir", failing_sync)

    at = _run_app()

    assert at.exception == []
    assert len(at.error) == 1
    assert "ドキュメントの同期に失敗しました" in at.error[0].value
    assert "disk full" in at.error[0].value
    # 同期失敗後もアプリはクラッシュせず、エージェント構築まで到達している
    assert "agent" in at.session_state


def test_startup_sync_with_failed_files_shows_warning_and_toast(monkeypatch):
    """異常系境界値: 追加/更新/削除に加え、一部ファイルの読み込みに失敗した場合、
    st.toastで変更件数、st.warningで失敗ファイル名一覧がそれぞれ表示され、
    アプリはクラッシュせず継続する
    （ingest.sync_data_dir()の戻り値仕様: added/updated/removed/failedの4キー。
    失敗ファイルがあるケースの回帰を防ぐために追加）。"""

    def partial_failure_sync(verbose=False):
        return {
            "added": ["good.txt"],
            "updated": [],
            "removed": [],
            "failed": ["bad.pdf", "broken.txt"],
        }

    monkeypatch.setattr(ingest, "sync_data_dir", partial_failure_sync)

    at = _run_app()

    assert at.exception == []
    assert at.error == []
    assert len(at.toast) == 1
    assert "追加1" in at.toast[0].value
    assert len(at.warning) == 1
    assert "bad.pdf" in at.warning[0].value
    assert "broken.txt" in at.warning[0].value
    # 失敗があってもエージェント構築（後続処理）は継続される
    assert "agent" in at.session_state


def test_resync_button_failure_shows_error(monkeypatch):
    """異常系: サイドバーの「🔄 data/ を再同期」ボタン押下時の同期失敗もカバーされる。

    このボタンは `st.expander("今すぐ強制的に再同期したい場合")` の中に
    移動したが、`AppTest`の`at.sidebar.button`はexpander内も含めてサイドバー配下の
    ボタンを再帰的に収集するため、取得方法自体は変更不要（folded状態でも
    要素ツリーには存在し、クリック操作も通常どおり可能）。
    """
    call_count = {"n": 0}

    def flaky_sync(verbose=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"added": [], "updated": [], "removed": [], "failed": []}
        raise RuntimeError("resync fail")

    monkeypatch.setattr(ingest, "sync_data_dir", flaky_sync)

    at = _run_app()
    assert at.error == []

    resync_button = next(b for b in at.sidebar.button if "再同期" in b.label)
    at = resync_button.click().run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "ドキュメントの同期に失敗しました" in at.error[0].value
    assert "resync fail" in at.error[0].value


# --- 2. チャット処理中の agent.invoke() 呼び出し ---


def test_chat_success_appends_history_and_shows_answer(monkeypatch):
    """正常系: agent.invoke() が成功すれば回答が表示され、会話履歴にも追加される。"""
    fake_agent = _FakeAgent(answer="これが回答です")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert at.error == []
    assert len(fake_agent.invoke_calls) == 1
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[0].content == "質問です"
    assert messages[1].content == "これが回答です"


def test_chat_invoke_failure_shows_ollama_message_when_provider_is_ollama(monkeypatch):
    """異常系: 使用中プロバイダがOllamaの場合、従来通り
    「Ollamaサーバーに接続できません」というメッセージが表示され、会話履歴には追加されない。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")
    fake_agent = _FakeAgent(exc=ConnectionError("connection refused"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "Ollamaサーバーに接続できません" in at.error[0].value
    assert "connection refused" in at.error[0].value
    # 例外時は履歴に追加されない（answerがNoneのまま後続処理がスキップされる）
    assert at.session_state["messages"] == []


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_chat_invoke_failure_shows_api_message_when_provider_is_cloud(monkeypatch, provider):
    """異常系: 使用中プロバイダがAnthropic/OpenAIの場合、
    Ollama決め打ちの誤ったメッセージではなく、APIへの接続失敗を示す汎用的な文言が表示される。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", provider)
    fake_agent = _FakeAgent(exc=RuntimeError("invalid api key"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "Ollamaサーバーに接続できません" not in at.error[0].value
    assert "APIへの接続に失敗しました" in at.error[0].value
    assert "invalid api key" in at.error[0].value
    assert at.session_state["messages"] == []


def test_chat_invoke_failure_shows_generic_fallback_when_provider_is_none(monkeypatch):
    """境界値: CURRENT_PROVIDER が未設定（None、想定外の状態）の場合、
    Ollama決め打ちの文言にもAPI決め打ちの文言にもならず、汎用的なフォールバック文言が
    表示される（_build_model()が何らかの理由で未実行・失敗した場合の保険）。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", None)
    fake_agent = _FakeAgent(exc=RuntimeError("unexpected"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "Ollamaサーバーに接続できません" not in at.error[0].value
    assert "APIへの接続に失敗しました" not in at.error[0].value
    assert "モデルへの接続に失敗しました" in at.error[0].value
    assert "unexpected" in at.error[0].value
    assert at.session_state["messages"] == []


def test_chat_invoke_failure_skips_auto_knowledge_save(monkeypatch):
    """異常系境界値: invoke失敗時は save_conversation が呼ばれない
    （＝data/conversations/への保存自体が発生しないため、後続の同期対象にもならない）。"""
    fake_agent = _FakeAgent(exc=RuntimeError("boom"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append(a))

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 起動時の1回のみ

    at.chat_input[0].set_value("質問です").run()

    assert save_calls == []
    assert sync_calls["n"] == 1  # invoke失敗のためsave_conversationが呼ばれず、同期も増えない


# --- 3. 会話ログ保存後の挙動（同一turn内での即時sync_data_dir呼び出しを廃止） ---


def test_post_chat_saves_conversation_without_immediate_resync(monkeypatch):
    """正常系: チャット応答後は save_conversation のみが呼ばれ、
    同じturn内では追加の sync_data_dir 呼び出しは発生しない。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append(a))

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 起動時の1回のみ

    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert at.error == []
    assert len(save_calls) == 1
    assert sync_calls["n"] == 1  # チャット応答直後には追加の同期は走らない
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[1].content == "回答"


def test_post_chat_signature_change_triggers_sync_on_next_run(monkeypatch):
    """正常系: 会話ログ保存でdata/内のファイルが増えたことを想定し、
    次回スクリプトが再実行されたタイミング（次のチャット送信等）で
    data_dir_signature() の変化を検知して sync_data_dir が呼ばれることを確認する。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: None)

    sig_holder = {"value": (1, 100.0)}
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: sig_holder["value"])

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 起動時の1回

    at = at.chat_input[0].set_value("質問1").run()
    assert sync_calls["n"] == 1  # 会話ログ保存直後はまだ同期されない（シグネチャ未変化）

    # 会話ログファイルが増えたことを想定してシグネチャを変化させる
    sig_holder["value"] = (2, 200.0)
    at = at.chat_input[0].set_value("質問2").run()

    assert at.exception == []
    assert sync_calls["n"] == 2  # 次回run()時にトップレベルの軽量チェックが検知して同期される


def test_post_chat_sync_failure_on_next_run_shows_error(monkeypatch):
    """異常系: 会話ログ保存後、次回run()時にトップレベルの同期が失敗した場合も
    st.errorが表示され、アプリはクラッシュしない。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: None)

    sig_holder = {"value": (1, 100.0)}
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: sig_holder["value"])

    call_count = {"n": 0}

    def flaky_sync(verbose=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"added": [], "updated": [], "removed": [], "failed": []}
        raise RuntimeError("resync fail on data dir change")

    monkeypatch.setattr(ingest, "sync_data_dir", flaky_sync)

    at = _run_app()
    at = at.chat_input[0].set_value("質問1").run()
    assert at.error == []

    sig_holder["value"] = (2, 200.0)
    at = at.chat_input[0].set_value("質問2").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "ドキュメントの同期に失敗しました" in at.error[0].value
    assert "resync fail on data dir change" in at.error[0].value
    # 同期失敗があっても直前のチャット応答自体は履歴に残っている
    messages = at.session_state["messages"]
    assert len(messages) == 4
    assert messages[-1].content == "回答"


def test_post_chat_save_conversation_not_called_when_auto_save_memory_disabled(monkeypatch):
    """境界値: 「今の会話を記憶として保存する」トグルOFFの場合は
    save_conversation が呼ばれず、それに伴う事後の同期も発生しない。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append(a))

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 起動時の1回

    toggle = at.sidebar.toggle[0]
    at = toggle.set_value(False).run()

    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert at.error == []
    assert save_calls == []
    assert sync_calls["n"] == 1  # 事後同期は呼ばれていない（data/自体に変化がないため）


# --- 4. サイドバーの記憶設定UI（会話ID表示とナレッジ化トグルの統合） ---


def test_memory_settings_expander_integrates_thread_info(monkeypatch):
    """正常系: 「🧠 記憶設定」expander内に、記憶保存トグル・保存件数・会話ID（補足情報）が
    まとめて表示され、生の会話IDそのものはユーザー向け説明文の主語になっていない
    （説明文キャプションの先頭は「今のチャット」であり、会話IDを含まない）。

    `AppTest`の`Block`（expanderを含む）は`.caption`/`.toggle`等の子要素アクセサを
    持ち、そのブロック配下のみを再帰的に収集する。ここではあえてサイドバー全体
    （`at.sidebar.caption`）ではなく取得した「🧠 記憶設定」expanderオブジェクト自身
    から辿ることで、これらの要素が実際にこの1つのexpander配下にまとまっていること
    （＝サイドバーの他の場所に分散していないこと）まで検証する。"""
    monkeypatch.setattr(memory, "new_thread_id", lambda: "abcd1234")
    monkeypatch.setattr(memory, "conversation_count", lambda thread_id: 3)

    at = _run_app()

    memory_expanders = [e for e in at.sidebar.expander if "記憶設定" in e.label]
    assert len(memory_expanders) == 1
    expander = memory_expanders[0]

    captions = [c.value for c in expander.caption]
    # 説明文（主語）に生の会話IDが含まれていないこと
    assert any("今のチャット" in c and "覚えておいて" in c for c in captions)
    assert not any(c.strip().startswith("abcd1234") for c in captions)
    # 保存件数・会話ID（補足情報としての表示）はそれぞれこのexpander配下で確認できる
    assert any("保存済みのやりとり: 3件" in c for c in captions)
    assert any("会話ID（内部識別用）: `abcd1234`" in c for c in captions)

    toggles = [t.label for t in expander.toggle]
    assert toggles == ["今の会話を記憶として保存する"]


# --- 5. トップレベルの軽量シグネチャチェック（data_dir_signature）そのもの ---


def test_rerun_without_signature_change_skips_resync(monkeypatch):
    """正常系: data_dir_signature() の値が前回run()と変わらなければ、
    再実行しても sync_data_dir は追加で呼ばれない（静かなno-op）。"""
    sig_holder = {"value": (1, 100.0)}
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: sig_holder["value"])

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 初回起動時は必ず同期される

    at = at.run()
    at = at.run()

    assert at.exception == []
    assert sync_calls["n"] == 1  # シグネチャが変化していないため追加の同期は起きない


def test_rerun_with_signature_change_triggers_resync(monkeypatch):
    """正常系: data_dir_signature() の値が前回run()から変化していれば、
    次回のrun()で sync_data_dir が呼ばれる。"""
    sig_holder = {"value": (1, 100.0)}
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: sig_holder["value"])

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 初回起動時

    # data/にファイルが増えた・変更されたことを想定してシグネチャを変化させる
    sig_holder["value"] = (2, 200.0)
    at = at.run()

    assert at.exception == []
    assert sync_calls["n"] == 2  # 変化を検知して同期が走る

    # シグネチャがさらに変化しなければ、その後は再び呼ばれない
    at = at.run()
    assert sync_calls["n"] == 2


# --- 6. プロダクト感のあるUI/UXへのリニューアル（画面表示文言） ---


def test_doclore_branding_title_and_tagline_are_displayed():
    """正常系: タイトルが新ブランド名「Doclore」、その下にキャッチコピーが
    表示される（`st.set_page_config` の page_title/page_icon 自体は要素ツリーに
    現れないメタ情報のため、静的検証は tests/test_theme_config.py 側で行う）。"""
    at = _run_app()

    assert at.exception == []
    assert len(at.title) == 1
    assert at.title[0].value == "📖 Doclore"
    assert any(m.value == "##### あなたの資料から、迷わず答えへ。" for m in at.markdown)


def test_sidebar_document_management_heading_has_folder_icon():
    """境界値: サイドバーの「ドキュメント管理」見出しに📂アイコンが付与され、
    かつ既存の見出しテキスト自体は変わっていない（絵文字の付け忘れ・文言の
    意図しない変更の両方を検知できるようにする）。"""
    at = _run_app()

    headings = [s.value for s in at.sidebar.subheader]
    assert headings == ["📂 ドキュメント管理"]


def test_chat_input_placeholder_uses_renewed_wording():
    """正常系: チャット入力欄のプレースホルダーが刷新後の文言になっている。"""
    at = _run_app()

    assert at.chat_input[0].placeholder == "資料について気になることを聞いてみましょう"


# --- 7. ファイルアップロード時の同名ファイル上書き防止 ---


def test_upload_new_file_is_saved_as_is_without_warning(tmp_path, monkeypatch):
    """正常系: data/に同名ファイルが存在しない通常のアップロードでは、
    元のファイル名のまま保存され、st.warningは表示されない。"""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)

    at = _run_app()
    at.file_uploader[0].set_value(("report.txt", b"new content", "text/plain"))
    at = at.run()

    assert at.exception == []
    assert at.error == []
    assert at.warning == []
    saved = data_dir / "report.txt"
    assert saved.exists()
    assert saved.read_bytes() == b"new content"


def test_upload_duplicate_filename_shows_warning_and_preserves_existing_file(tmp_path, monkeypatch):
    """異常系境界値: data/に既に同名ファイルが存在する場合、
    既存ファイルを上書きせず連番サフィックス付きの別名で保存し、
    st.warningでリネームされた旨をまとめて通知する。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "report.txt").write_bytes(b"existing content")
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)

    at = _run_app()
    at.file_uploader[0].set_value(("report.txt", b"new content", "text/plain"))
    at = at.run()

    assert at.exception == []
    assert at.error == []
    assert len(at.warning) == 1
    assert "report.txt" in at.warning[0].value
    assert "report (2).txt" in at.warning[0].value
    # 既存ファイルは上書きされず内容が保たれている
    assert (data_dir / "report.txt").read_bytes() == b"existing content"
    # 新しいファイルは連番サフィックス付きの別名で保存される
    assert (data_dir / "report (2).txt").read_bytes() == b"new content"


def test_upload_same_name_files_in_one_batch_shows_warning(tmp_path, monkeypatch):
    """境界値: data/上にはまだ存在しなくても、同一アップロードバッチ内に
    同名ファイルが複数含まれる場合は2件目以降が別名で保存され、st.warningが表示される。"""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)

    at = _run_app()
    at.file_uploader[0].set_value(
        [
            ("dup.txt", b"first content", "text/plain"),
            ("dup.txt", b"second content", "text/plain"),
        ]
    )
    at = at.run()

    assert at.exception == []
    assert at.error == []
    assert len(at.warning) == 1
    assert "dup.txt" in at.warning[0].value
    assert "dup (2).txt" in at.warning[0].value
    assert (data_dir / "dup.txt").read_bytes() == b"first content"
    assert (data_dir / "dup (2).txt").read_bytes() == b"second content"


def test_upload_invalid_filename_shows_error_without_warning(tmp_path, monkeypatch):
    """異常系境界値: resolve_upload_dest()がNoneを返す場合（パストラバーサルの疑い等、
    実際の再現には st.file_uploader 自身の拡張子制限を回避する必要があるため、
    ここでは resolve_upload_dest() 自体を直接フェイクにして分岐だけを検証する）、
    該当ファイルはst.errorでスキップ表示され、リネーム扱いにならないためst.warningは
    表示されない。"""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "resolve_upload_dest", lambda filename, taken_paths=None: None)

    at = _run_app()
    at.file_uploader[0].set_value(("evil.txt", b"content", "text/plain"))
    at = at.run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "不正なファイル名のためスキップしました: evil.txt" in at.error[0].value
    assert at.warning == []


# --- 8. 読み込み失敗ファイルの警告永続化・自動リトライ ---


def test_failed_sync_files_warning_persists_and_retries_until_fixed(monkeypatch):
    """異常系: 読み込みに失敗したファイルがdata/内に残っている間は、
    data_dir_signature()（ファイル数+最新mtimeのみを見る軽量判定）が変化しなくても、
    次回以降のスクリプト再実行のたびに自動でsync_data_dirが再試行され、
    警告も消えずに表示され続ける。修正されて同期が成功すれば警告は消え、
    以降の再実行では余計な同期が走らなくなる（元の状態に戻る）。"""
    # data/自体は一切変化していない想定でシグネチャを固定する
    sig_holder = {"value": (1, 100.0)}
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: sig_holder["value"])

    call_count = {"n": 0}

    def flaky_sync(verbose=False):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return {"added": [], "updated": [], "removed": [], "failed": ["broken.pdf"]}
        return {"added": ["broken.pdf"], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", flaky_sync)

    at = _run_app()
    assert call_count["n"] == 1
    assert len(at.warning) == 1
    assert "broken.pdf" in at.warning[0].value

    # シグネチャは変化していないが、失敗ファイルが残っているため自動的に再試行される
    at = at.run()
    assert call_count["n"] == 2
    assert len(at.warning) == 1
    assert "broken.pdf" in at.warning[0].value

    # ファイルが修正され、今回は同期に成功したとする
    at = at.run()
    assert call_count["n"] == 3
    assert at.warning == []

    # 成功後はシグネチャが保存されるため、以降のrun()では余計な同期は走らない
    at = at.run()
    assert call_count["n"] == 3


def test_sync_success_without_failed_files_updates_signature_and_no_warning(monkeypatch):
    """正常系: 失敗ファイルが無い同期では、従来通りdata_dir_signatureがセッションに
    保存され警告は表示されない。次回run()でシグネチャが変化していなければ、
    余計な同期も走らない。"""
    sig_holder = {"value": (1, 100.0)}
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: sig_holder["value"])

    call_count = {"n": 0}

    def counting_sync(verbose=False):
        call_count["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()

    assert at.exception == []
    assert at.warning == []
    assert at.session_state["data_dir_signature"] == sig_holder["value"]
    assert at.session_state["failed_sync_files"] == []

    at = at.run()
    assert call_count["n"] == 1  # シグネチャ更新済みのため2回目のrun()では同期されない
    assert at.warning == []


def test_failed_sync_files_keep_signature_unset_until_recovered(monkeypatch):
    """異常系境界値: 失敗ファイルが残っている間はdata_dir_signatureがセッションに
    保存されず、failed_sync_filesにも失敗ファイル名がそのまま保持され続ける。
    ファイルが修正されて同期が成功すると、両方とも正常時の状態にクリアされる。"""
    sig_holder = {"value": (5, 500.0)}
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: sig_holder["value"])

    call_count = {"n": 0}

    def flaky_sync(verbose=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"added": [], "updated": [], "removed": [], "failed": ["corrupt.pdf"]}
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", flaky_sync)

    at = _run_app()
    assert "data_dir_signature" not in at.session_state
    assert at.session_state["failed_sync_files"] == ["corrupt.pdf"]

    # シグネチャ未保存のため、data/自体が無変化でも次回run()で自動的に再試行される
    at = at.run()
    assert call_count["n"] == 2
    assert at.session_state["data_dir_signature"] == sig_holder["value"]
    assert at.session_state["failed_sync_files"] == []
    assert at.warning == []
