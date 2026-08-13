import sys
from pathlib import Path

# リポジトリ直下の setup.py を import できるよう、examples/ の1つ上の階層を sys.path に追加する。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setup import model  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402


# messages = [
#     SystemMessage(content="Translate the following from English to Italian."),
#     HumanMessage(content="'Hello, how are you?'"),
# ]

# model.invoke(messages)

# print(model.invoke(messages))


# # 以下、同等のコード例
# print(model.invoke("Hello"))
# print(model.invoke([{"role": "user", "content": "Hello"}]))
# print(model.invoke([HumanMessage("Hello")]))
# # ストリーミング応答の処理例
# for token in model.stream(messages):
#     print(token.content, end="|")


from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

system_template = "Translate the following from English into {language}."

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)


def main() -> None:
    prompt = prompt_template.invoke({"language": "Italian", "text": "Hello, how are you?"})
    print(prompt)
    print(prompt.to_messages())

    response = model.invoke(prompt)
    print(response.content)


if __name__ == "__main__":
    main()
