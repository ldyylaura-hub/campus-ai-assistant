from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def format_docs(docs):
    return "docs"

try:
    print("Creating RunnableLambda...")
    format_docs_runnable = RunnableLambda(lambda x: format_docs(x["context"]))
    print(f"RunnableLambda created: {format_docs_runnable}, type: {type(format_docs_runnable)}")

    print("Calling RunnablePassthrough.assign...")
    assign_step = RunnablePassthrough.assign(context=format_docs_runnable)
    print(f"Assign step created: {assign_step}")

except Exception as e:
    import traceback
    traceback.print_exc()
