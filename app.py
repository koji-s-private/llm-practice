"""
ローカルドキュメントに質問できるRAGチャットアプリ「Doclore」（Streamlit）。

起動:
    python -m streamlit run app.py

data/ フォルダにファイルを置く（またはサイドバーからアップロードする）だけでOK。
data/ フォルダの変更は、ページの操作（リロード・チャットの送信など、Streamlitが
スクリプトを再実行するタイミング）のたびに軽量な変更検知で自動的に検知され、
裏側で自動的にベクトルDBへ反映されます（手動での再同期は基本不要。即時性が必要な
場合のフォールバックとして、サイドバーの折りたたみ内に手動の再同期ボタンもあります）。
Google Driveとの連携（設定方法はdocs/google-drive-setup.md参照）を設定済みの場合、
同じ折りたたみ内の「🔄 Google Driveと同期」ボタンから手動でオンデマンド同期できます。

さらに、チャットでの質問・回答も自動で data/conversations/<会話スレッドID>/ に保存され、
「このスレッド」の次回以降の質問（別セッション・別タブでも同じスレッドを開けば）の
回答材料として使われます。サイドバーの「🆕 新しい会話を始める」を押すと新しいスレッドIDが
発行され、以前の会話ログは検索対象から外れる（＝無関係な過去の会話が回答に混ざらない）ようになります。
サイドバーの「💬 過去の会話」から過去のスレッドを選んで再開することもでき、
選択したスレッドの会話履歴がチャット画面に復元されます。

保存先はすべてこのプロジェクト内のローカルディスク（data/ と chroma_db/）のみで、
このアプリ自身が外部・クラウドへ追加送信することはありません。
"""

from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from streamlit.delta_generator import DeltaGenerator

import google_drive_sync
import setup

# 会話履歴のトークン数ウィンドウイング（history_utils.py）・参照元表示の整形
# （source_formatting.py）は app.py と api/main.py の両方から使う共通ロジックのため
# 切り出している。テストが従来通り app._windowed_history 等の名前で参照できるよう、
# "as 同名" で明示的に再エクスポートする（ruffのunused-import誤検知を防ぐ）。
from history_utils import _API_PROVIDER_HISTORY_TOKENS as _API_PROVIDER_HISTORY_TOKENS
from history_utils import _FALLBACK_HISTORY_TOKENS as _FALLBACK_HISTORY_TOKENS
from history_utils import _OLLAMA_CONTEXT_MARGIN_TOKENS as _OLLAMA_CONTEXT_MARGIN_TOKENS
from history_utils import _OLLAMA_MIN_HISTORY_TOKENS as _OLLAMA_MIN_HISTORY_TOKENS
from history_utils import _history_token_budget as _history_token_budget
from history_utils import _windowed_history as _windowed_history
from ingest import (
    DATA_DIR,
    add_single_conversation_file,
    data_dir_signature,
    delete_indexed_file,
    list_indexed_files,
    resolve_upload_dest,
    sync_data_dir,
)
from memory import conversation_count, list_threads, load_conversation, new_thread_id, save_conversation
from rag_chain import build_agent
from source_formatting import format_snippet as _format_snippet
from source_formatting import format_source_label as _format_source_label

# 配色・フォントは .streamlit/config.toml のカスタムテーマで設定している。
st.set_page_config(
    page_title="Doclore | ドキュメントAIアシスタント",
    page_icon="📖",
    layout="centered",
)
st.title("📖 Doclore")
st.markdown("##### あなたの資料から、迷わず答えへ。")
st.caption("data/ フォルダにファイルを置くと自動でDBに反映され、AIエージェントが検索しながら回答します。")


def _sync_and_report(spinner_text: str, warning_slot: DeltaGenerator | None = None) -> None:
    """data/全体を差分同期し、結果をトースト・警告バナーに反映する。

    warning_slot（トップレベルで確保済みのst.empty()）を渡すと、次のスクリプト再実行を
    待たずに同じターン内で警告バナーへ即時反映できる（起動時の呼び出しではスロット確保前
    のため渡さない）。
    """
    try:
        with st.spinner(spinner_text):
            result = sync_data_dir(verbose=False)
    except Exception as e:
        st.error(f"ドキュメントの同期に失敗しました。時間をおいて再度お試しください。（詳細: {e}）")
        # 失敗時はシグネチャを更新しない。次回もトップレベルの軽量チェックが再同期を試みる。
        return
    if any(result.values()):
        st.toast(
            f"DBを更新しました（追加{len(result['added'])} / "
            f"更新{len(result['updated'])} / 削除{len(result['removed'])}）",
            icon="✅",
        )
    # data_dir_signatureはファイルが存在する限り変化しないため、失敗ファイル一覧を
    # セッションに保持しておかないと警告表示がこの1回のスクリプト実行でしか出せない。
    st.session_state.failed_sync_files = result["failed"]
    if warning_slot is not None:
        _show_failed_sync_files_warning(warning_slot)
    if result["failed"]:
        # 失敗ファイルが残っている間はシグネチャを更新せず、次回も再同期・再試行させる。
        return
    # 手動の再同期ボタン・アップロード時もこの関数を通るため、成功後にシグネチャを
    # 更新しておくことで直後の軽量チェックによる無駄な二重同期を防ぐ。
    st.session_state.data_dir_signature = data_dir_signature()


def _sync_google_drive_and_report(warning_slot: DeltaGenerator | None = None) -> None:
    """Google Driveの内容をdata/google_drive/にミラーし、続けてDBへ反映して結果を通知する。

    GOOGLE_DRIVE_FOLDER_ID未設定時、sync_google_drive_files()は例外を出さず全キー空リストを
    返す仕様のため（google_drive_sync.py参照）、それと「設定済みだが変更なし」を区別せず
    未設定寄りの案内で共通化する（変更なしの場合に誤情報にはならないため実害は無い）。
    認証情報ファイルが無い場合はRuntimeErrorが送出されるため、他の失敗と分けてエラー表示する。
    """
    try:
        with st.spinner("Google Driveと同期中..."):
            drive_result = google_drive_sync.sync_google_drive_files(verbose=False)
    except RuntimeError as e:
        st.error(f"Google Drive連携の認証情報が見つかりません。（詳細: {e}）")
        return
    except Exception as e:
        st.error(f"Google Driveとの同期に失敗しました。時間をおいて再度お試しください。（詳細: {e}）")
        return

    if not any(drive_result.values()):
        st.info(
            "Google Drive連携が未設定、または同期対象の変更はありませんでした。"
            "連携の設定方法は docs/google-drive-setup.md を参照してください。"
        )
    else:
        st.toast(
            f"Google Driveの内容を同期しました（追加{len(drive_result['added'])} / "
            f"更新{len(drive_result['updated'])} / 削除{len(drive_result['removed'])} / "
            f"スキップ{len(drive_result['skipped'])}）",
            icon="✅",
        )

    _sync_and_report("data/ をベクトルDBに反映中...", warning_slot)


def _sync_saved_conversation(path: Path, warning_slot: DeltaGenerator | None = None) -> None:
    """保存したばかりの会話ログ1件だけを、data/全件を走査せずその場で軽量にDB反映する。

    トップレベルの軽量シグネチャチェック（sync_data_dir）に任せると、data/配下の
    ファイル数に比例して毎ターンの走査コストが増え続けるため、add_single_conversation_file()で
    対象1件だけを処理する。失敗時はシグネチャを更新せず、次回の全件差分同期に再試行を委ねる。
    """
    try:
        status = add_single_conversation_file(path)
    except Exception as e:
        st.error(f"会話ログの保存処理でDBへの反映に失敗しました。（詳細: {e}）")
        return
    if status == "failed":
        try:
            name = str(path.relative_to(DATA_DIR))
        except ValueError:
            # DATA_DIR配下でないパス（テスト用のフェイクパス等）でもクラッシュしないための保険。
            name = str(path)
        failed_sync_files = st.session_state.get("failed_sync_files") or []
        if name not in failed_sync_files:
            st.session_state.failed_sync_files = [*failed_sync_files, name]
        _show_failed_sync_files_warning(warning_slot)
        return
    # このファイル追加でdata/の内容が変わるため、次回rerun時に無駄な
    # sync_data_dir()呼び出しが走らないようシグネチャも更新しておく。
    st.session_state.data_dir_signature = data_dir_signature()


def _show_failed_sync_files_warning(container: DeltaGenerator | None = None) -> None:
    """読み込みに失敗したファイルの警告を、同期が成功するまで毎回のスクリプト実行で表示し続ける。

    st.warningはそのスクリプト実行の描画にしか残らないため、セッションに保持した
    失敗ファイル一覧を毎回参照して描画する。containerを渡すとそのスロットへ上書き描画し、
    失敗ファイルが0件になった場合はcontainer.empty()で古い警告をクリアする。
    """
    target = container if container is not None else st
    failed = st.session_state.get("failed_sync_files")
    if not failed:
        if container is not None:
            container.empty()
        return
    target.warning(
        "以下のファイルは読み込みに失敗したため、DBへの反映がスキップされています"
        "（破損・パスワード付き・不正なエンコーディング等の可能性があります）。"
        "data/ から修正・削除すると自動的に再試行されます:\n" + "\n".join(f"- {name}" for name in failed)
    )


def _format_invoke_error_message(e: Exception) -> str:
    """agent.invoke()/agent.stream()失敗時のエラーメッセージを、実際に使用中のプロバイダに応じて出し分ける。

    setup.py の _build_model() はOllama→Anthropic→OpenAIの順にフォールバックするため、
    setup.CURRENT_PROVIDER を見ずに固定メッセージにすると、実際とは異なるプロバイダの
    トラブルシューティングにユーザーを誤誘導してしまう。
    """
    if setup.CURRENT_PROVIDER == "ollama":
        message = str(e).lower()
        if "model" in message and "not found" in message:
            # 起動時チェック（setup._ollama_model_pulled()）をすり抜けたケースや、
            # 起動後に別プロセスでモデルが削除された場合の保険。
            cause = (
                f"モデル '{setup.OLLAMA_MODEL}' が見つかりません。"
                f"'ollama pull {setup.OLLAMA_MODEL}' を実行するか、"
                "OLLAMA_MODEL の設定を見直してください。"
            )
        else:
            cause = "Ollamaサーバーに接続できません。起動しているか確認してください。"
    elif setup.CURRENT_PROVIDER in ("anthropic", "openai"):
        cause = "APIへの接続に失敗しました。APIキーやネットワーク接続、レート制限などをご確認ください。"
    else:
        # CURRENT_PROVIDER未設定（想定外のケース）でも汎用的な文言でフォールバックする
        cause = "モデルへの接続に失敗しました。"
    return f"回答の生成に失敗しました。{cause}（詳細: {e}）"


def _show_provider_fallback_warning() -> None:
    """Ollama（無料・ローカル）が使えず有料APIにフォールバックしている場合、起動直後に警告バナーを表示する。

    ユーザーが気づかないうちに課金対象のAPIが使われ続けることを防ぐため、
    setup.CURRENT_PROVIDERが"ollama"以外になっているスクリプト実行では毎回表示する。
    """
    if setup.CURRENT_PROVIDER == "ollama" or setup.CURRENT_PROVIDER is None:
        return
    reason = setup.CURRENT_PROVIDER_FALLBACK_REASON
    reason_text = f"（理由: {reason}）" if reason else ""
    st.warning(
        f"⚠️ ローカル無料実行(Ollama)が利用できないため、有料API（{setup.CURRENT_PROVIDER}）を使用しています。"
        f"{reason_text}"
    )


def _build_agent_safely(thread_id: str):
    """build_agent()を例外から保護する共通ヘルパー。

    現状のbuild_agent()はネットワーク呼び出しを伴わないため失敗しにくいが、
    将来モデルの疎通確認などが追加された場合に備え、失敗してもアプリ全体を
    クラッシュさせずエラー表示に留める。失敗時はNoneを返す。
    """
    try:
        return build_agent(thread_id)
    except Exception as e:
        st.error(f"RAGエージェントの初期化に失敗しました。時間をおいて再度お試しください。（詳細: {e}）")
        return None


def _start_new_chat() -> None:
    st.session_state.thread_id = new_thread_id()
    st.session_state.messages = []
    st.session_state.agent = _build_agent_safely(st.session_state.thread_id)
    # スレッド選択のselectboxが古いスレッドの選択状態を保持したままだと、次の再実行時に
    # 「選択値 != 新しいthread_id」と誤判定されて選択スレッドへ引き戻されてしまうためリセットする。
    st.session_state.pop("thread_selector", None)


def _format_message_timestamp(timestamp: datetime | None) -> str | None:
    """チャット画面に添えるタイムスタンプ表示を作る。timestampが無い場合はNoneを返す。

    同じ日なら"14:32"、日をまたぐ場合は日付も添えて"07/29 14:32"のように表示する。
    """
    if timestamp is None:
        return None
    if timestamp.date() == datetime.now().date():
        return timestamp.strftime("%H:%M")
    return timestamp.strftime("%m/%d %H:%M")


def _format_thread_label(thread: dict) -> str:
    """過去スレッド選択UI用に、作成日時と最初の質問の要約を組み合わせたラベルを作る。"""
    timestamp = thread["created_at"].strftime("%Y-%m-%d %H:%M")
    snippet = _format_snippet(thread["first_question"], limit=24) if thread["first_question"] else "(質問内容なし)"
    return f"{timestamp}｜{snippet}（{thread['count']}件）"


def _conversation_to_markdown(messages: list, thread_id: str) -> str:
    """現在のスレッドの会話履歴（HumanMessage/AIMessage）をエクスポート用のMarkdownに整形する。"""
    lines = [
        "# 会話ログ",
        "",
        f"- スレッドID: {thread_id}",
        f"- エクスポート日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for i, message in enumerate(messages, start=1):
        role = "質問" if isinstance(message, HumanMessage) else "回答"
        lines.append(f"## {role} {(i + 1) // 2}")
        lines.append("")
        lines.append(message.content)
        lines.append("")
    return "\n".join(lines)


def _render_indexed_file_list() -> None:
    """インデックス済みファイルの一覧を表示し、各ファイルの削除ボタンから個別削除できるようにする。

    誤操作でファイルを消してしまわないよう、削除は「削除ボタン→確認ボタン」の2段階にする。
    確認待ちの状態はセッションに保持し、削除完了・キャンセルのいずれかで解除する。
    """
    indexed_files = list_indexed_files()
    if not indexed_files:
        st.caption("インデックス済みのファイルはまだありません。")
        return

    st.caption(f"インデックス済みファイル: {len(indexed_files)}件")
    for file_info in indexed_files:
        name = file_info["name"]
        pending_key = f"pending_delete_{name}"
        col_label, col_button = st.columns([5, 1])
        col_label.markdown(f"📄 {name}　`{file_info['chunk_count']}チャンク`")
        if col_button.button("🗑️", key=f"delete_button_{name}", help=f"{name} を削除"):
            st.session_state[pending_key] = True

        if st.session_state.get(pending_key):
            st.warning(f"「{name}」を削除します。この操作は取り消せません。よろしいですか？")
            col_confirm, col_cancel = st.columns(2)
            if col_confirm.button("削除する", key=f"confirm_delete_{name}", type="primary"):
                deleted = delete_indexed_file(name)
                st.session_state.pop(pending_key, None)
                if deleted:
                    _sync_and_report(f"{name} を削除中...")
                    st.rerun()
                else:
                    # 削除失敗（対象ファイルが既に無い等）はUIの状態を変える必要が無いため、
                    # rerunせずこのスクリプト実行内でst.errorをそのまま表示する。
                    st.error(f"「{name}」の削除に失敗しました（既に削除されている可能性があります）。")
            if col_cancel.button("キャンセル", key=f"cancel_delete_{name}"):
                st.session_state.pop(pending_key, None)
                st.rerun()


def _render_answer_provenance(sources: list) -> None:
    """回答がドキュメント根拠か一般知識かのバッジと、参照元expanderを回答直後・過去ターン再描画の両方で表示する。

    sourcesが空か否かをそのまま判定に使う（save_conversation()のis_fallback判定と同じ考え方）ため、
    additional_kwargs["sources"]としてメッセージ本体に保持済みのsourcesを渡せば、
    再描画時も追加の状態を持たずに同じ判定結果を再現できる。
    """
    if not sources:
        st.caption("🧠 一般知識による回答（ドキュメントに該当情報なし）")
        return
    st.caption("🔍 ドキュメントに基づく回答")
    with st.expander("参照した箇所を見る"):
        for i, doc in enumerate(sources, start=1):
            label = _format_source_label(doc.metadata)
            st.markdown(f"**[{i}] {label}**")
            st.text(_format_snippet(doc.page_content))


def _switch_thread(thread_id: str) -> None:
    """選択された過去スレッドに切り替え、そのスレッドの会話履歴をチャット画面に復元する。"""
    st.session_state.thread_id = thread_id
    messages = []
    for turn in load_conversation(thread_id):
        timestamp_kwargs = {"timestamp": turn["created_at"]}
        if turn["question"]:
            messages.append(HumanMessage(content=turn["question"], additional_kwargs=timestamp_kwargs))
        if turn["answer"]:
            messages.append(AIMessage(content=turn["answer"], additional_kwargs=timestamp_kwargs))
    st.session_state.messages = messages
    st.session_state.agent = _build_agent_safely(thread_id)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()

# Streamlitは操作のたびにスクリプト全体を再実行する仕様なので、軽量シグネチャ
# （ファイル数+最新mtime、内容は読まない）で前回値と比較し、data/の変更
# （リロード時の外部編集・会話ログ保存の両方）だけを検知して同期する。
current_data_dir_signature = data_dir_signature()
if st.session_state.get("data_dir_signature") != current_data_dir_signature:
    _sync_and_report("data/ をベクトルDBに同期中...")

# 前回までの同期で読み込みに失敗したファイルが残っている場合、このスクリプト実行でも
# 警告を表示し続ける（同期が呼ばれなかった場合でも、直前の失敗状態を毎回描画するため）。
# プレースホルダーとして確保しておくことで、この後の会話保存（_sync_saved_conversation）・
# 再同期ボタン・アップロード時（いずれも_sync_and_report経由）が同じターン中に成功/失敗
# しても、新規要素を追加せずこのスロットへ上書きで即座に反映できる。
failed_sync_warning_slot = st.empty()
_show_failed_sync_files_warning(failed_sync_warning_slot)

# エージェント自体はdata/の変更とは独立して一度だけ構築すればよい
# （検索ツールはベクトルストアを都度クエリするため、同期結果は再構築なしで自動的に反映される）。
if "agent" not in st.session_state:
    with st.spinner("RAGエージェントを準備中..."):
        st.session_state.agent = _build_agent_safely(st.session_state.thread_id)

if "messages" not in st.session_state:
    st.session_state.messages = []  # 表示・履歴用（HumanMessage / AIMessage）

if "auto_save_memory" not in st.session_state:
    st.session_state.auto_save_memory = True  # 会話の自動ナレッジ化（デフォルトON）

if "processed_upload_ids" not in st.session_state:
    # st.file_uploaderの値はファイルを明示的に取り除くかリロードするまで保持され続けるため、
    # 無関係な操作での再実行でも同じファイルが含まれ続ける。保存・DB反映済みのfile_idを
    # ここに記録し、再実行のたびに重複保存・再インデックスされないようにする。
    st.session_state.processed_upload_ids = set()

with st.sidebar:
    _show_provider_fallback_warning()

    if st.button("🆕 新しい会話を始める", use_container_width=True):
        _start_new_chat()
        # st.rerun()はその場でスクリプト実行を打ち切るため、直前のst.error()の描画内容も
        # 次の描画で失われてしまう。エージェント構築に失敗した場合はrerunせず、この回の
        # 実行内でエラーメッセージがそのまま表示され続けるようにする。
        if st.session_state.agent is not None:
            st.rerun()

    st.divider()
    st.subheader("💬 過去の会話")
    past_threads = list_threads()
    if not past_threads:
        st.caption("まだ保存された会話スレッドはありません。")
    else:
        thread_labels = {t["thread_id"]: _format_thread_label(t) for t in past_threads}
        selected_thread_id = st.selectbox(
            "過去のスレッドを選んで再開",
            options=list(thread_labels.keys()),
            format_func=lambda tid: thread_labels[tid],
            index=None,
            placeholder="スレッドを選択...",
            key="thread_selector",
            label_visibility="collapsed",
        )
        # 選択値が現在表示中のスレッドと異なる場合のみ切り替える。同じ場合はスキップし、
        # 選択操作以外の理由での再実行（他のウィジェット操作等）で毎回再構築されないようにする。
        if selected_thread_id and selected_thread_id != st.session_state.thread_id:
            _switch_thread(selected_thread_id)
            # 新しい会話を始める場合と同様、エージェント構築に失敗した場合はrerunせず
            # st.error()の描画をこの回の実行内に残す。
            if st.session_state.agent is not None:
                st.rerun()

    st.divider()
    # 「会話ID」という生のID文字列を主語にした表示ではなく、「今の会話を記憶に残すか」
    # という1つの設定としてまとめて提示する。session_stateのキー名（auto_save_memory）
    # は既存の保存処理ロジックとの互換性のため変更しない。
    with st.expander("🧠 記憶設定", expanded=False):
        st.caption("今のチャットでのやりとりを覚えておいて、次回以降の質問の回答材料に使うかどうかを設定します。")
        st.session_state.auto_save_memory = st.toggle(
            "今の会話を記憶として保存する",
            value=st.session_state.auto_save_memory,
            help=(
                "ONの場合、やりとりを data/conversations/ にローカル保存し、"
                "この会話スレッド内での以降の質問の回答材料にします"
                "（別スレッドの会話には混ざりません）。外部・クラウドへの追加送信は一切行いません。"
            ),
        )
        st.caption(f"この会話で保存済みのやりとり: {conversation_count(st.session_state.thread_id)}件")
        st.caption(f"会話ID（内部識別用）: `{st.session_state.thread_id}`")

    st.divider()
    st.subheader("📥 会話のエクスポート")
    st.caption("現在表示中のスレッドの質問・回答をMarkdownファイルとしてダウンロードします。")
    has_messages = bool(st.session_state.messages)
    export_filename = f"conversation_{st.session_state.thread_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    st.download_button(
        "📥 この会話をダウンロード",
        data=_conversation_to_markdown(st.session_state.messages, st.session_state.thread_id) if has_messages else "",
        file_name=export_filename,
        mime="text/markdown",
        disabled=not has_messages,
        use_container_width=True,
    )

    st.divider()
    st.subheader("📂 ドキュメント管理")
    st.caption(
        "data/ フォルダの変更はページの操作（リロード・会話など）のたびに自動で検知され、"
        "裏側で自動的にDBへ反映されます。"
    )
    _render_indexed_file_list()

    # 自動検知はファイル数+最新mtimeによる近似的な判定のため、理論上は「同じmtime・
    # 同じサイズのまま中身だけ入れ替わる」ような極めて稀なケースを取りこぼす可能性がある。
    # 即時性・確実性が必要な場合のフォールバック手段として、目立たない場所に残しておく。
    with st.expander("今すぐ強制的に再同期したい場合"):
        if st.button("🔄 data/ を再同期"):
            _sync_and_report("再同期中...", failed_sync_warning_slot)
        if st.button("🔄 Google Driveと同期"):
            _sync_google_drive_and_report(failed_sync_warning_slot)

    st.caption("ファイルをアップロードすると自動で data/ に保存・DB反映されます。")
    uploaded_files = st.file_uploader(
        "ファイルを追加",
        type=["pdf", "txt", "md", "docx", "csv", "xlsx", "xls", "pptx", "html", "htm"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    # 既に保存・DB反映済みのfile_idを持つファイルは除外する。これにより、アップロード欄を
    # 操作していない再実行（チャット送信・他ボタン押下など）では新規保存対象が0件になり、
    # 以降の保存処理・_sync_and_report()自体が実質no-opになる。
    new_uploaded_files = [f for f in (uploaded_files or []) if f.file_id not in st.session_state.processed_upload_ids]
    if new_uploaded_files:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # 同名ファイルが既にある場合はresolve_upload_dest()が連番サフィックス付きの
        # パスを返すので、元のファイル名と異なる場合はリネームされたとみなし警告表示する。
        saved_paths: set[Path] = set()
        renamed = []
        for f in new_uploaded_files:
            dest = resolve_upload_dest(f.name, taken_paths=saved_paths)
            # st.file_uploader(type=[...])はサーバー側でも拡張子を検証するため、
            # 通常この分岐には到達しないが、resolve_upload_dest()自体は汎用関数であり将来
            # 呼び出し元が増える可能性もあるため多重防御として残している。
            if dest is None:
                st.error(f"不正なファイル名のためスキップしました: {f.name}")
                st.session_state.processed_upload_ids.add(f.file_id)
                continue
            if dest.name != f.name:
                renamed.append((f.name, dest.name))
            dest.write_bytes(f.getvalue())
            saved_paths.add(dest)
            st.session_state.processed_upload_ids.add(f.file_id)
        if renamed:
            st.warning(
                "同名のファイルが既に存在したため、既存ファイルを上書きせず別名で保存しました:\n"
                + "\n".join(f"- {old} → {new}" for old, new in renamed)
            )
        _sync_and_report("アップロードされたファイルを取り込み中...", failed_sync_warning_slot)

for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        timestamp_label = _format_message_timestamp(message.additional_kwargs.get("timestamp"))
        if timestamp_label:
            st.caption(timestamp_label)
        st.markdown(message.content)
        if isinstance(message, AIMessage):
            # additional_kwargsはAPI送信時には未知キーとして無視されるため、ここに参照元を
            # 積んでおいても後続のagent.stream()への影響なくセッション内で保持できる。
            _render_answer_provenance(message.additional_kwargs.get("sources") or [])

user_input = st.chat_input("資料について気になることを聞いてみましょう")

if user_input:
    turn_timestamp = datetime.now()
    with st.chat_message("user"):
        st.caption(_format_message_timestamp(turn_timestamp))
        st.markdown(user_input)

    answer = None
    sources: list = []
    with st.chat_message("assistant"):
        st.caption(_format_message_timestamp(turn_timestamp))
        if st.session_state.agent is None:
            st.error(
                "RAGエージェントが利用できないため、回答を生成できません。ページを再読み込みして再度お試しください。"
            )
            st.stop()
        try:
            # 検索中であることを示すプレースホルダー。最初の回答トークンが届いた時点で消す
            # （ツール呼び出し中は回答本文のトークンが生成されないため、その間の待機を可視化する）。
            status_placeholder = st.empty()
            status_placeholder.markdown("🔍 検索して回答を考え中...")
            # st.write_stream() が内部的に生成するプレースホルダーは呼び出し元から
            # 参照できず、ストリーム途中で例外が起きた場合に描画済みの部分テキストを
            # クリアできない。自前でプレースホルダーを持つことで、except節から
            # 明示的にクリアできるようにする。
            answer_placeholder = st.empty()

            def _stream_answer():
                """agentのストリーミング出力を逐次yieldしつつ、参照元ドキュメントをsourcesへ蓄積する。

                ToolMessage（検索ツールの実行結果）は回答本文ではないため除外し、artifactだけを
                sourcesに蓄積する。AIMessageChunk.content はプロバイダによって型が異なる
                （str、またはAnthropicのcontent blocks list）ため、素朴なisinstance判定ではなく
                text系ブロックを結合済みの .text プロパティでテキストを取り出す。
                """
                first_token = True
                seen_source_keys: set = set()
                for chunk, _metadata in st.session_state.agent.stream(
                    {"messages": _windowed_history(st.session_state.messages) + [HumanMessage(content=user_input)]},
                    stream_mode="messages",
                ):
                    if isinstance(chunk, ToolMessage):
                        if getattr(chunk, "artifact", None):
                            # 1ターン中に複数回検索されて同じチャンクが重複ヒットすることがあるため、
                            # 既出チャンクを除外する。page_contentもキーに含めるのは、pageを持たない
                            # .txt/.md等ではsource/thread_idだけでは同一ファイル内の別チャンクを
                            # 区別できないため。
                            for doc in chunk.artifact:
                                key = (
                                    doc.metadata.get("source"),
                                    doc.metadata.get("page"),
                                    doc.metadata.get("thread_id"),
                                    doc.page_content,
                                )
                                if key in seen_source_keys:
                                    continue
                                seen_source_keys.add(key)
                                sources.append(doc)
                        continue
                    text = getattr(chunk, "text", "")
                    if text:
                        if first_token:
                            status_placeholder.empty()
                            first_token = False
                        yield text

            # st.write_stream()ではなく手動で蓄積・描画する。理由は上記の
            # answer_placeholderのコメントの通り、例外発生時に部分描画を
            # クリアできるようにするため。
            accumulated_answer = ""
            for text in _stream_answer():
                accumulated_answer += text
                answer_placeholder.markdown(accumulated_answer)
            answer = accumulated_answer
            # 回答トークンが1つも届かなかった場合（ツール呼び出しのみで終わった等）に備え、
            # プレースホルダーが残っていれば消す。
            status_placeholder.empty()

            _render_answer_provenance(sources)
        except Exception as e:
            status_placeholder.empty()
            # ストリーム途中（一部チャンクをyield済み）で例外が発生した場合に、
            # 描画済みの部分的な回答テキストを画面に残さずクリアする。
            answer_placeholder.empty()
            answer = None
            st.error(_format_invoke_error_message(e))

    if answer is not None:
        st.session_state.messages.append(
            HumanMessage(content=user_input, additional_kwargs={"timestamp": turn_timestamp})
        )
        # 参照元・タイムスタンプは再描画ループでも表示できるよう、additional_kwargsに載せて
        # メッセージ本体と一緒に保持する。
        st.session_state.messages.append(
            AIMessage(content=answer, additional_kwargs={"sources": sources, "timestamp": turn_timestamp})
        )

        # 会話を自動でナレッジ化（このスレッド専用でローカル保存し、全件走査するsync_data_dir()
        # ではなく保存した1ファイルだけをその場でDB反映）。sourcesが空＝根拠なしの一般知識回答
        # なのでis_fallbackとして記録し、以降の検索対象から除外できるようにする。
        if st.session_state.auto_save_memory:
            saved_path = save_conversation(user_input, answer, st.session_state.thread_id, is_fallback=not sources)
            _sync_saved_conversation(saved_path, failed_sync_warning_slot)
