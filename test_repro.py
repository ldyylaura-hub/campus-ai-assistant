from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.runnables.base import coerce_to_runnable

def format_docs(docs):
    return "foo"

print("Creating RunnableLambda...")
rl = RunnableLambda(lambda x: format_docs(x))
print(f"RL: {rl}, type: {type(rl)}")

print("Testing coerce_to_runnable...")
try:
    c = coerce_to_runnable(rl)
    print(f"Coerced: {c}")
except Exception as e:
    print(f"Coerce failed: {e}")

print("Testing RunnableParallel direct...")
try:
    rp = RunnableParallel({"context": rl})
    print(f"RP: {rp}")
except Exception as e:
    print(f"RP failed: {e}")

print("Testing RunnablePassthrough.assign...")
try:
    assign = RunnablePassthrough.assign(context=rl)
    print(f"Assign: {assign}")
except Exception as e:
    print(f"Assign failed: {e}")
