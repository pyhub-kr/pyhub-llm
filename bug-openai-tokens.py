from pyhub.llm import OpenAILLM, UpstageLLM

llm = OpenAILLM()
# llm = UpstageLLM()

reply = llm.ask(
    "우울해서 빵을 샀어.",
    choices=["기쁨", "슬픔", "분노", "불안", "무기력함"],
)
print(repr(reply))
print(reply.choice)  # "슬픔"
print(reply.choice_index)  # 1
