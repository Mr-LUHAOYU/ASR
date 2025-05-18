import gradio as gr
import numpy as np
from matplotlib import pyplot as plt
from evaluator import Evaluator
import shutil


class Page(object):
    def __init__(self, model_zoo, tempfile='temp.wav'):
        self.tempfile = tempfile
        self._upload_init()
        self._model_init()
        self._extractor = Evaluator(model_zoo)

    def _upload_init(self):
        gr.Markdown("## 上传音频文件")
        self.audio = gr.Audio(type="filepath", label="上传WAV文件")

    def _model_init(self):
        gr.Markdown('## 请选择模型')
        self.model = gr.Dropdown(
            ["CNN", "LSTM", "Combine", "SVM"],
            label="选择模型类型",
            value='Combine',
            interactive=True
        )
        gr.Markdown('## 情感分析结果')
        self.emotion = gr.Markdown('尚未输入语音')
        gr.Button('开始分析').click(
            self.handle,
            inputs=[self.audio, self.model],
            outputs=[self.emotion],
        )

    def handle(self, audio, model):
        print('uploading...')
        self.upload(audio)
        print('done')
        print('calculating emotion')
        emotion = self.get_emotion(model)
        print('emotion:', emotion)
        return emotion
        # self.plot(self.imgs)

    def upload(self, audio):
        shutil.copyfile(audio, self.tempfile)

    def get_emotion(self, model):
        self._extractor.set_model(model)
        emotion = self._extractor(audio=self.tempfile)
        return str(emotion)

    @property
    def imgs(self):
        return 1, 2, 3

    def plot(self, *figs):
        gr.Markdown("## 数据可视化")

        # 创建图表
        with gr.Row():
            with gr.Column():
                for fig in figs:
                    gr.Matplotlib(fig)
