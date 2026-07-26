from setup import model
from langchain_core.messages import HumanMessage, SystemMessage


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


from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}."

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "Hello, how are you?"})
print(prompt)
print(prompt.to_messages())

response = model.invoke(prompt)
print(response.content)
