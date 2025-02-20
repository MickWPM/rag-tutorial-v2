import gradio as gr

import query_data

def greet(query_text):
    response = query_data.query_rag(query_text)
    return response

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()