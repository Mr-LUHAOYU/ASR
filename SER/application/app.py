import gradio as gr
from base import Page

# 创建界面
with gr.Blocks(title="语音情感分析") as demo:
    gr.Markdown("# 语音情感分析系统")
    Page('f:/ASR/SER/zoo')

# 启动应用
if __name__ == "__main__":
    demo.launch()
