from litgpt import LLM
import time

model = "microsoft/phi-2"
print(f"Loading model {model}")
llm = LLM.load(model)
text = llm.generate("Fix the spelling: Every fall, the family goes to the mountains.")
print(text)
# Corrected Sentence: Every fall, the family goes to the mountains.