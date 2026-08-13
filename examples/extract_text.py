import sys
from pathlib import Path
from typing import List, Optional

# リポジトリ直下の setup.py を import できるよう、examples/ の1つ上の階層を sys.path に追加する。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # noqa: E402
from langchain_core.utils.function_calling import tool_example_to_messages  # noqa: E402

from setup import model  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

llm = model

class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an 'optional' -- this allows the model to decline to extract it!
    # 2. Each field has a 'description' -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(
        default=None,
        description="The name of the person.",
    )
    hair_color: Optional[str] = Field(
        default=None,
        description="the color of the person's hair is known",
    )
    height_in_meters: Optional[float] = Field(
        default=None,
        description="Height measured in meters",
    )

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account
#    (e.g., include metadata about the document from which the text was extracted.)

class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm."
            "Only extract relevant information from the text."
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

# structured_llm = llm.with_structured_output(schema=Person)

# text = "Alan Smith is 6 feet tall and has blond hair."
# prompt = prompt_template.invoke({"text": text})
# # structured_llm.invoke(prompt)
# print(structured_llm.invoke(prompt))


# messages = [
#     {
#         "role": "user", "content": "2 🦆 2"
#     },
#     {
#         "role": "assistant", "content": "4"
#     },
#     {
#         "role": "user", "content": "2 🦆 3"
#     },
#     {
#         "role": "assistant", "content": "5"
#     },
#     {
#         "role": "user", "content": "3 🦆 4"
#     },
# ]

# response = llm.invoke(messages)
# print(response.content)


# examples = [
#     (
#         "The ocean is vast and blue. It's more than 20,000 feet deep.",
#         Data(people=[]),
#     ),
#     (
#         "Fiona traveled far from France to Spain.",
#         Data(people=[
#             Person(name="Fiona", hair_color=None, height_in_meters=None),
#         ]),
#     )
# ]

# for txt, tool_call in examples:
#     if tool_call.people:
#         # This final message is optional for some providers
#         ai_response = "Detected people."
#     else:
#         ai_response = "Detected no people."
#     messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))

# for message in messages:
#     message.pretty_print()

message_no_extraction = {
    "role": "user",
    "content": "The solar system is large, but earth has only one moon.",
}


def main() -> None:
    text = "My name is Jeff, my hair is black and i am 6 feet tall. " "Anna has the same color hair as me."
    prompt = prompt_template.invoke({"text": text})

    structured_llm = llm.with_structured_output(schema=Data)
    # structured_llm.invoke(prompt)
    print(structured_llm.invoke(prompt))

    messages = []
    structured_llm = llm.with_structured_output(schema=Data)
    # structured_llm.invoke([message_no_extraction])
    # print(structured_llm.invoke([message_no_extraction]))
    print(structured_llm.invoke(messages + [message_no_extraction]))


if __name__ == "__main__":
    main()
