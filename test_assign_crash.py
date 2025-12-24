from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.runnables.base import Runnable

def format_docs(docs):
    return "formatted"

format_docs_runnable = RunnableLambda(lambda x: format_docs(x["context"]))

print("--- Reproduction Test ---")
try:
    # Mimic exactly what RunnablePassthrough.assign does
    kwargs = {"context": format_docs_runnable}
    # Note: The traceback showed RunnableParallel[Dict[str, Any]](kwargs)
    # We simulate passing dict as positional arg
    rp = RunnableParallel(kwargs)
    print("RunnableParallel(kwargs) created:", rp)
    
    assign = RunnablePassthrough.assign(context=format_docs_runnable)
    print("RunnablePassthrough.assign created:", assign)
except Exception as e:
    print("Caught expected error:", e)
    import traceback
    traceback.print_exc()

print("\n--- Workaround Test ---")
try:
    def merge_context(x):
        # Emulate assign
        return {**x, "context": format_docs(x["context"])}
    
    workaround_step = RunnableLambda(merge_context)
    print("Workaround step created:", workaround_step)
    
    # Test invoke
    res = workaround_step.invoke({"context": ["doc1"], "input": "query"})
    print("Workaround invoke result:", res)
except Exception as e:
    print("Workaround failed:", e)
    traceback.print_exc()
