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
            ["MLP", "LSTM", "Combine"],
            label="选择模型类型",
            value='Combine',
            interactive=True
        )
        self.dataset = gr.Dropdown(
            ["SAVEE", 'Ravdess'],
            label="选择预训练的数据",
            value='SAVEE',
            interactive=True
        )
        self.noise = gr.Radio(
            ['带噪训练', '无噪训练'],
            label='无噪训练',
            type='index'
        )
        self.mfcc = gr.Dropdown(
            [f'{mfcc * 13}*{delta}'
                for mfcc in range(1, 4)
                for delta in range(1, 4)],
            label='选择MFCC采样参数',
            value=f'39*3',
            interactive=True
        )
        gr.Markdown('## 情感分析结果')
        self.emotion = gr.Markdown('尚未输入语音')
        gr.Button('开始分析').click(
            self.handle,
            inputs=[self.audio, self.model, self.dataset, self.mfcc],
            outputs=[self.emotion],
        )

    def handle(self, audio, model, dataset, mfcc):
        print('uploading...')
        self.upload(audio)
        print('done')
        print('calculating emotion')
        emotion = self.get_emotion(model, dataset, mfcc)
        print('# emotion:', emotion)
        return emotion

    def upload(self, audio):
        shutil.copyfile(audio, self.tempfile)

    def get_emotion(self, model, dataset, mfcc):
        self._extractor.set_model(model, dataset, mfcc)
        emotion = self._extractor(audio=self.tempfile, mfcc=mfcc)
        return str(emotion)

