# https://doupoa.site/archives/595

import os
import sys
import wave
import numpy as np

sys.path.append('./')
sys.path.append('./third_party/Matcha-TTS')
current_dir = os.path.dirname(os.path.abspath(__file__))

import time

import json
import torch
import torchaudio
from tqdm import tqdm

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

with open('./examples/tts/audio/speaker_data.json', 'r', encoding='utf-8') as file:
    speaker_dict = json.load(file)

# 将第三方库Matcha-TTS的路径添加到系统路径中
sys.path.append('third_party/Matcha-TTS')

# 记录开始时间
start = time.time()
# 初始化CosyVoice2模型，指定预训练模型路径，不加载jit和trt模型，使用fp32

cosyvoice = CosyVoice2('iic/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

# 设置最大音量
max_val = 0.8
# 设置说话人名称
speaker = 'leijun'
# 设置说话人信息文件的路径
spk2info_path = current_dir + '/spk2info.pt'

# 加载说话人信息文件
spk2info = torch.load(spk2info_path, map_location=cosyvoice.frontend.device)

print("spk2info", spk2info.keys())


# 定义一个文本到语音的函数，参数包括文本内容、是否流式处理、语速和是否使用文本前端处理
def tts_sft(tts_text, speaker_info: dict, stream=False, speed=1.0, text_frontend=True):
    '''
    参数：
        tts_text：要合成的文本
        speaker：说话人音频特征
        stream：是否流式处理
        speed：语速
        text_frontend：是否使用文本前端处理

    返回值：
        合成后的音频
    '''
    # 使用tqdm库来显示进度条，对文本进行标准化处理并分割
    for i in tqdm(cosyvoice.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
        # 提取文本的token和长度
        tts_text_token, tts_text_token_len = cosyvoice.frontend._extract_text_token(i)
        # 提取提示文本的token和长度
        prompt_text_token, prompt_text_token_len = cosyvoice.frontend._extract_text_token(
            speaker_dict[speaker]["prompt_text"])
        # 获取说话人的语音token长度，并转换为torch张量，移动到指定设备
        speech_token_len = torch.tensor([speaker_info['speech_token'].shape[1]], dtype=torch.int32).to(
            cosyvoice.frontend.device)
        # 获取说话人的语音特征长度，并转换为torch张量，移动到指定设备
        speech_feat_len = torch.tensor([speaker_info['speech_feat'].shape[1]], dtype=torch.int32).to(
            cosyvoice.frontend.device)
        # 构建模型输入字典，包括文本、文本长度、提示文本、提示文本长度、LLM提示语音token、LLM提示语音token长度、流提示语音token、流提示语音token长度、提示语音特征、提示语音特征长度、LLM嵌入和流嵌入
        model_input = {
            'text': tts_text_token, 'text_len': tts_text_token_len,
            'prompt_text': prompt_text_token,
            'prompt_text_len': prompt_text_token_len,
            'llm_prompt_speech_token': speaker_info['speech_token'],
            'llm_prompt_speech_token_len': speech_token_len,
            'flow_prompt_speech_token': speaker_info['speech_token'],
            'flow_prompt_speech_token_len': speech_token_len,
            'prompt_speech_feat': speaker_info['speech_feat'], 'prompt_speech_feat_len': speech_feat_len,
            'llm_embedding': speaker_info['embedding'], 'flow_embedding': speaker_info['embedding']
        }
        # 使用模型进行文本到语音的转换，并迭代输出结果
        for model_output in cosyvoice.model.tts(**model_input, stream=stream, speed=speed):
            yield model_output



tts_text_list = ["收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。", ]

# 遍历文本列表
for text in tts_text_list:
    # 记录开始时间
    start = time.time()
    # 遍历每个文本的生成结果
    for i, j in enumerate(tts_sft(text, speaker_info=spk2info[speaker], stream=False, speed=1.2)):
        # 保存生成的语音到文件，文件名包含文本的前四个字符
        print('打印处理时间:', i, time.time() - start)
        output_path = current_dir + '/audio/test_{}_{}.wav'.format(i, speaker)
        
        # torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)


        audio_data = (j['tts_speech'].numpy() * 32767).astype(np.int16)
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(cosyvoice.sample_rate)
            wf.writeframes(audio_data.tobytes())
        
    # 打印处理时间
    print('打印处理时间:', time.time() - start)


