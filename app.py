"""
ローカルドキュメントに質問できるRAGチャットアプリ（Streamlit / 最低限のUI）。

起動:
    python -m streamlit run app.py

data/ フォルダにファイルを置く（またはサイドバーからアップロードする）だけでOK。
起動時・サイドバーの「再同期」ボタン・アップロード時に自動でベクトルDBへ反映されます。

さらに、チャットでの質問・回答も自動で data/conversations/<会話スレッドID>/ に保存され、
「このスレッド」の次回以降の質問（別セッション・別タブでも同じスレッドを開けば）の
回答材料として使われます。サイドバーの「🆕 新しい会話を始める」を押すと新しいスレッドIDが
発行され、以前の会話ログは検索対象から外れる（＝無関係な過去の会話が回答に混ざらない）ようになります。

保存先はすべてこのプロジェクト内のローカルディスク（data/ と chroma_db/）のみで、
このアプリ自身が外部・クラウドへ追加送信することはありません。
"""
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ingest import DATA_DIR, sync_data_dir
from memory import conversation_count, new_thread_id, save_conversation
from rag_chain import build_agent

st.set_page_config(page_title="llm-practice RAGチャット", page_icon="📚")
st.title("📚 ローカルドキュメントQ&A")
st.caption("data/ フォルダにファイルを置くと自動でDBに反映され、AIエージェントが検索しながら回答します。")


def _sync_and_report(spinner_text: str) -> None:
    with st.spinner(spinner_text):
        result = sync_data_dir(verbose=False)
    if any(result.values()):
        st.toast(
            f"DBを更新しました（追加{len(result['added'])} / "
            f"更新{len(result['updated'])} / 削除{len(result['removed'])}）",
            icon="✅",
        )


def _start_new_chat() -> None:
    st.session_state.thread_id = new_thread_id()
    st.session_state.messages = []
    st.session_state.agent = build_agent(st.session_state.thread_id)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()

# 初回アクセス時にdata/の内容をベクトルDBへ自動同期
if "agent" not in st.session_state:
    _sync_and_report("data/ をベクトルDBに同期中...")
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
    st.caption(
        f"会話ID: `{st.session_state.thread_id}`（このIDの会話ログだけが、"
        "この会話の回答材料として検索されます）"
    )

    st.divider()
    st.subheader("ドキュメント管理")
    st.caption("data/ フォルダにファイルを追加・削除したら押してください。")
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
        for f in uploaded_files:
            (DATA_DIR / f.name).write_bytes(f.getvalue())
        _sync_and_report("アップロードされたファイルを取り込み中...")

    st.divider()
    st.subheader("会話の自動ナレッジ化")
    st.session_state.auto_save_memory = st.toggle(
        "質問・回答を自動で保存する",
        value=st.session_state.auto_save_memory,
        help=(
            "ONの場合、やりとりを data/conversations/ にローカル保存し、"
            "この会話スレッド内での以降の質問の回答材料にします"
            "（別スレッドの会話には混ざりません）。外部・クラウドへの追加送信は一切行いません。"
        ),
    )
    st.caption(f"このスレッドの保存済み会話: {conversation_count(st.session_state.thread_id)}件")

# 過去の会話を再描画
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

user_input = st.chat_input("ドキュメントについて質問してください")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("検索して回答を考え中..."):
            result = st.session_state.agent.invoke(
                {"messages": st.session_state.messages + [HumanMessage(content=user_input)]}
            )
            new_messages = result["messages"]
            answer = new_messages[-1].content
            st.markdown(answer)

            # ツール呼び出しで取得した参照元ドキュメントを表示
            sources = []
            for m in new_messages:
                if isinstance(m, ToolMessage) and getattr(m, "artifact", None):
                    sources.extend(m.artifact)

            if sources:
                with st.expander("参照した箇所を見る"):
                    for i, doc in enumerate(sources, start=1):
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page")
                        label = Path(source).name if source != "unknown" else source
                        if page is not None:
                            label += f"（p.{page + 1}）"
                        st.markdown(f"**[{i}] {label}**")
                        st.text(doc.page_content[:300] + "...")

    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(AIMessage(content=answer))

    # 会話を自動でナレッジ化（このスレッド専用としてローカル保存 → 即座にDB反映）
    if st.session_state.auto_save_memory:
        save_conversation(user_input, answer, st.session_state.thread_id)
        sync_data_dir(verbose=False)
