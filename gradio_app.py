import gradio as gr
def predict(text):
    return text.upper()
gr.Interface(fn=predict, inputs="text", outputs="text").launch()