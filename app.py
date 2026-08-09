"""
ローカルドキュメントに質問できるRAGチャットアプリ「Doclore」（Streamlit）。

起動:
    python -m streamlit run app.py

data/ フォルダにファイルを置く（またはサイドバーからアップロードする）だけでOK。
data/ フォルダの変更は、ページの操作（リロード・チャットの送信など、Streamlitが
スクリプトを再実行するタイミング）のたびに軽量な変更検知で自動的に検知され、
裏側で自動的にベクトルDBへ反映されます（手動での再同期は基本不要。即時性が必要な
場合のフォールバックとして、サイドバーの折りたたみ内に手動の再同期ボタンもあります）。

さらに、チャットでの質問・回答も自動で data/conversations/<会話スレッドID>/ に保存され、
「このスレッド」の次回以降の質問（別セッション・別タブでも同じスレッドを開けば）の
回答材料として使われます。サイドバーの「🆕 新しい会話を始める」を押すと新しいスレッドIDが
発行され、以前の会話ログは検索対象から外れる（＝無関係な過去の会話が回答に混ざらない）ようになります。
サイドバーの「💬 過去の会話」から過去のスレッドを選んで再開することもでき、
選択したスレッドの会話履歴がチャット画面に復元されます。

保存先はすべてこのプロジェクト内のローカルディスク（data/ と chroma_db/）のみで、
このアプリ自身が外部・クラウドへ追加送信することはありません。
"""

from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import setup
from ingest import DATA_DIR, data_dir_signature, resolve_upload_dest, sync_data_dir
from memory import conversation_count, list_threads, load_conversation, new_thread_id, save_conversation
from rag_chain import GLOBAL_THREAD_ID, build_agent

# 配色・フォントは .streamlit/config.toml のカスタムテーマで設定している。
st.set_page_config(
    page_title="Doclore | ドキュメントAIアシスタント",
    page_icon="📖",
    layout="centered",
)
st.title("📖 Doclore")
st.markdown("##### あなたの資料から、迷わず答えへ。")
st.caption("data/ フォルダにファイルを置くと自動でDBに反映され、AIエージェントが検索しながら回答します。")


def _format_snippet(text: str, limit: int = 300) -> str:
    """参照元プレビュー用に本文を整形する。

    単純に先頭limit文字で切ると、文や単語の途中で不自然に切れてしまい、
    limit未満の短いテキストにまで"..."が付いてしまう問題があった。
    - limit文字以内に収まる場合はそのまま返し、"..."は付けない。
    - limitを超える場合は、句点・改行などの区切り文字のうち最も末尾に近いものを探し、
      そこで区切る（区切り位置が手前すぎる場合は意味が無いのでlimitの半分より
      後ろにある場合のみ採用し、見つからなければ直近の空白で単語の途中を避けて切る）。
    """
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped

    truncated = stripped[:limit]
    break_chars = "。\n！？!?"
    best_pos = max((truncated.rfind(ch) for ch in break_chars), default=-1)
    if best_pos >= limit // 2:
        truncated = truncated[: best_pos + 1]
    else:
        space_pos = truncated.rfind(" ")
        if space_pos >= limit // 2:
            truncated = truncated[:space_pos]

    return truncated.rstrip() + "..."


def _format_source_label(metadata: dict) -> str:
    """参照元ドキュメントのメタデータから表示用ラベルを組み立てる。

    - source: ファイルパス → ファイル名のみを表示
    - thread_id: 会話ログ由来のチャンクにのみ付与される（GLOBAL_THREAD_IDは
      全スレッド共通ドキュメントを表すため対象外）。付与されている場合は
      「会話ログ（スレッド: xxx）」であることが分かるように先頭に付ける
    - page: PDFのページ番号（0始まり）があれば「（p.N）」を末尾に付ける
    """
    source = metadata.get("source", "unknown")
    page = metadata.get("page")
    thread_id = metadata.get("thread_id")

    label = Path(source).name if source != "unknown" else source
    if thread_id and thread_id != GLOBAL_THREAD_ID:
        label = f"会話ログ（スレッド: {thread_id}） - {label}"
    if page is not None:
        label += f"（p.{page + 1}）"
    return label


def _sync_and_report(spinner_text: str) -> None:
    try:
        with st.spinner(spinner_text):
            result = sync_data_dir(verbose=False)
    except Exception as e:
        st.error(f"ドキュメントの同期に失敗しました。時間をおいて再度お試しください。（詳細: {e}）")
        # 失敗時はシグネチャを更新しない。次回の再実行時もdata/の内容は
        # 「未同期」のままとみなされ、トップレベルの軽量チェックが再度同期を試みる。
        return
    if any(result.values()):
        st.toast(
            f"DBを更新しました（追加{len(result['added'])} / "
            f"更新{len(result['updated'])} / 削除{len(result['removed'])}）",
            icon="✅",
        )
    # 読み込みに失敗したファイル名一覧をセッションに保持しておく。data_dir_signatureは
    # ファイルが存在する限り変化しないため、この情報がないと警告表示がこの1回の
    # スクリプト実行でしか出ず、次にユーザーが操作した瞬間に消えてしまう。
    st.session_state.failed_sync_files = result["failed"]
    if result["failed"]:
        # 失敗ファイルが残っている間はシグネチャを更新しない。これにより
        # 「data/の内容自体は変化していない」場合でも、トップレベルの軽量チェックが
        # 引き続き「未同期」と判定し、次回のスクリプト再実行時に自動的に再同期・
        # 再試行される（対象ファイルが修正・削除されるまでリトライが続く）。
        return
    # 同期成功後の最新シグネチャを保存しておく。これにより、この直後にトップレベルの
    # 軽量チェックが再実行されても「変更なし」と判定され、無駄な二重同期が走らない
    # （手動の再同期ボタン・アップロード時の呼び出しでも共通してこの関数を通るため）。
    st.session_state.data_dir_signature = data_dir_signature()


def _show_failed_sync_files_warning() -> None:
    """読み込みに失敗したファイルの警告を、同期が成功するまで毎回のスクリプト実行で表示し続ける。

    st.warningはそのスクリプト実行の描画にしか残らないため、_sync_and_report()内で
    一度呼ぶだけでは次の画面操作（別のチャット送信・ボタン押下等）で消えてしまう。
    ここでセッションに保持した失敗ファイル一覧を毎回参照して描画することで、
    ユーザーがファイルを修正・削除して同期が成功するまで警告が残り続けるようにする。
    """
    failed = st.session_state.get("failed_sync_files")
    if not failed:
        return
    st.warning(
        "以下のファイルは読み込みに失敗したため、DBへの反映がスキップされています"
        "（破損・パスワード付き・不正なエンコーディング等の可能性があります）。"
        "data/ から修正・削除すると自動的に再試行されます:\n" + "\n".join(f"- {name}" for name in failed)
    )


def _format_invoke_error_message(e: Exception) -> str:
    """agent.invoke()/agent.stream()失敗時のエラーメッセージを、実際に使用中のプロバイダに応じて出し分ける。

    setup.py の _build_model() はOllama→Anthropic→OpenAIの順にフォールバックするため、
    Claude/OpenAIで動作しているセッションでは「Ollamaサーバーに接続できません」という
    固定メッセージは原因と食い違い、ユーザーを誤った方向（Ollamaの起動確認）に
    誘導してしまう。setup.CURRENT_PROVIDER を参照し、実際の使用プロバイダに即した文言にする。
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


def _start_new_chat() -> None:
    st.session_state.thread_id = new_thread_id()
    st.session_state.messages = []
    st.session_state.agent = build_agent(st.session_state.thread_id)
    # スレッド選択のselectboxが古いスレッドの選択状態を保持したままだと、次の再実行時に
    # 「選択値 != 新しいthread_id」と誤判定されて選択スレッドへ引き戻されてしまうためリセットする。
    st.session_state.pop("thread_selector", None)


def _format_thread_label(thread: dict) -> str:
    """過去スレッド選択UI用に、作成日時と最初の質問の要約を組み合わせたラベルを作る。"""
    timestamp = thread["created_at"].strftime("%Y-%m-%d %H:%M")
    snippet = _format_snippet(thread["first_question"], limit=24) if thread["first_question"] else "(質問内容なし)"
    return f"{timestamp}｜{snippet}（{thread['count']}件）"


def _switch_thread(thread_id: str) -> None:
    """選択された過去スレッドに切り替え、そのスレッドの会話履歴をチャット画面に復元する。"""
    st.session_state.thread_id = thread_id
    messages = []
    for turn in load_conversation(thread_id):
        if turn["question"]:
            messages.append(HumanMessage(content=turn["question"]))
        if turn["answer"]:
            messages.append(AIMessage(content=turn["answer"]))
    st.session_state.messages = messages
    st.session_state.agent = build_agent(thread_id)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()

# data/ の変更検知: Streamlitはユーザー操作（チャット送信・ボタン押下・
# トグル操作等）のたびにこのスクリプト全体を再実行する仕様なので、トップレベルで
# 「ファイル数+最新mtime」だけの軽量シグネチャ（data_dir_signature、内容の読み込みや
# 埋め込み処理は一切しない）を毎回計算し、前回値と比較する。これにより、
# 1) ページのリロード時（アプリ外からdata/を直接編集して戻ってきた場合を含む）
# 2) チャットの往復が続いたタイミング（会話ログの保存でdata/内のファイルが増えるため）
# の両方を、この1つの仕組みだけで自動検知できる（差分が無ければ何もしない静かなno-op）。
current_data_dir_signature = data_dir_signature()
if st.session_state.get("data_dir_signature") != current_data_dir_signature:
    _sync_and_report("data/ をベクトルDBに同期中...")

# 前回までの同期で読み込みに失敗したファイルが残っている場合、このスクリプト実行でも
# 警告を表示し続ける（同期が呼ばれなかった場合でも、直前の失敗状態を毎回描画するため）。
_show_failed_sync_files_warning()

# エージェント自体はdata/の変更とは独立して一度だけ構築すればよい
# （検索ツールはベクトルストアを都度クエリするため、同期結果は再構築なしで自動的に反映される）。
if "agent" not in st.session_state:
    with st.spinner("RAGエージェントを準備中..."):
        st.session_state.agent = build_agent(st.session_state.thread_id)

if "messages" not in st.session_state:
    st.session_state.messages = []  # 表示・履歴用（HumanMessage / AIMessage）

if "auto_save_memory" not in st.session_state:
    st.session_state.auto_save_memory = True  # 会話の自動ナレッジ化（デフォルトON）

with st.sidebar:
    if st.button("🆕 新しい会話を始める", use_container_width=True):
        _start_new_chat()
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
    st.subheader("📂 ドキュメント管理")
    st.caption(
        "data/ フォルダの変更はページの操作（リロード・会話など）のたびに自動で検知され、"
        "裏側で自動的にDBへ反映されます。"
    )
    # 自動検知はファイル数+最新mtimeによる近似的な判定のため、理論上は「同じmtime・
    # 同じサイズのまま中身だけ入れ替わる」ような極めて稀なケースを取りこぼす可能性がある。
    # 即時性・確実性が必要な場合のフォールバック手段として、目立たない場所に残しておく。
    with st.expander("今すぐ強制的に再同期したい場合"):
        if st.button("🔄 data/ を再同期"):
            _sync_and_report("再同期中...")

    st.caption("ファイルをアップロードすると自動で data/ に保存・DB反映されます。")
    uploaded_files = st.file_uploader(
        "ファイルを追加",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # data/に同名ファイルが既にある場合、または同一バッチ内に同名ファイルが
        # 複数含まれる場合に、無警告で上書きされないようにする。
        # resolve_upload_dest()が連番サフィックス付きの空いているパスを返すので、
        # 元のファイル名と異なる場合はリネームされたとみなしてまとめて警告表示する。
        saved_paths: set[Path] = set()
        renamed = []
        for f in uploaded_files:
            dest = resolve_upload_dest(f.name, taken_paths=saved_paths)
            if dest is None:
                st.error(f"不正なファイル名のためスキップしました: {f.name}")
                continue
            if dest.name != f.name:
                renamed.append((f.name, dest.name))
            dest.write_bytes(f.getvalue())
            saved_paths.add(dest)
        if renamed:
            st.warning(
                "同名のファイルが既に存在したため、既存ファイルを上書きせず別名で保存しました:\n"
                + "\n".join(f"- {old} → {new}" for old, new in renamed)
            )
        _sync_and_report("アップロードされたファイルを取り込み中...")

# 過去の会話を再描画
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

user_input = st.chat_input("資料について気になることを聞いてみましょう")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    answer = None
    sources: list = []
    with st.chat_message("assistant"):
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

                stream_mode="messages" は (メッセージチャンク, メタデータ) のタプルを順に返す。
                ToolMessage（検索ツールの実行結果）のcontentはツールの生の検索結果テキストであり、
                回答本文として表示すべきではないため除外し、artifact（取得ドキュメント）だけを
                sourcesに蓄積する。回答本文（AIMessageChunk）のテキストのみをyieldする。

                AIMessageChunk.content はプロバイダによって型が異なる（OpenAIは常にstr、
                Anthropicはtoolsをbindしている場合 [{"type": "text", "text": "..."}] のような
                content blocksのlistで返る）。BaseMessage.text プロパティはstr/listいずれの形式でも
                text系ブロックのみを結合した文字列を返してくれるため、素朴なisinstance(content, str)判定
                ではなくこちらを使い、プロバイダによらず本文を取りこぼさないようにする。
                """
                first_token = True
                for chunk, _metadata in st.session_state.agent.stream(
                    {"messages": st.session_state.messages + [HumanMessage(content=user_input)]},
                    stream_mode="messages",
                ):
                    if isinstance(chunk, ToolMessage):
                        if getattr(chunk, "artifact", None):
                            sources.extend(chunk.artifact)
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

            # ツール呼び出しで取得した参照元ドキュメントを表示
            if sources:
                with st.expander("参照した箇所を見る"):
                    for i, doc in enumerate(sources, start=1):
                        label = _format_source_label(doc.metadata)
                        st.markdown(f"**[{i}] {label}**")
                        st.text(_format_snippet(doc.page_content))
        except Exception as e:
            status_placeholder.empty()
            # ストリーム途中（一部チャンクをyield済み）で例外が発生した場合に、
            # 描画済みの部分的な回答テキストを画面に残さずクリアする。
            answer_placeholder.empty()
            answer = None
            st.error(_format_invoke_error_message(e))

    if answer is not None:
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.session_state.messages.append(AIMessage(content=answer))

        # 会話を自動でナレッジ化（このスレッド専用としてローカル保存）。
        # 保存したログファイルのDB反映は、この場ですぐには行わない。
        # 次回このスクリプトが再実行されたタイミング（次の会話・ボタン操作・リロード等）で、
        # トップレベルの軽量シグネチャチェックがファイル数の増加を検知し、自動的に同期される
        # （チャット1往復ごとに毎回フル同期する実装より軽量）。
        # sourcesが空＝retrieve_contextが関連文書を1件も見つけられず一般知識で回答した
        # ケースなので、is_fallbackとして記録し、以降の検索対象から除外できるようにする。
        if st.session_state.auto_save_memory:
            save_conversation(user_input, answer, st.session_state.thread_id, is_fallback=not sources)
