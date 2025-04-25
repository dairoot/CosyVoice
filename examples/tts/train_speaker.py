# https://doupoa.site/archives/595

import os

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append('./')
sys.path.append('./third_party/Matcha-TTS')

import time

import torch
import torchaudio
from tqdm import tqdm
import json
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

with open('./examples/tts/audio/speaker_data.json', 'r', encoding='utf-8') as file:
    speaker_dict = json.load(file)

print(speaker_dict)

# 将第三方库Matcha-TTS的路径添加到系统路径中
sys.path.append('third_party/Matcha-TTS')

# 记录开始时间
start = time.time()
# 初始化CosyVoice2模型，指定预训练模型路径，不加载jit和trt模型，使用fp32

cosyvoice = CosyVoice2('iic/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# 设置最大音量
max_val = 0.8
# 设置说话人名称
speaker = 'leijun'
# 设置说话人信息文件的路径
spk2info_path = current_dir + '/spk2info.pt'

# 加载16kHz的提示语音
prompt_speech_16k = load_wav(speaker_dict[speaker]["audio"], 16000)

# 如果说话人信息文件存在，则加载
if os.path.exists(spk2info_path):
    spk2info = torch.load(
        spk2info_path, map_location=cosyvoice.frontend.device)
else:
    spk2info = {}

print("spk2info", spk2info.keys())

if speaker not in spk2info:
    print("train speaker: {}".format(speaker))
    # 获取音色embedding
    embedding = cosyvoice.frontend._extract_spk_embedding(prompt_speech_16k)
    # 获取语音特征
    prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=cosyvoice.sample_rate)(
        prompt_speech_16k)
    speech_feat, speech_feat_len = cosyvoice.frontend._extract_speech_feat(prompt_speech_resample)
    # 获取语音token
    speech_token, speech_token_len = cosyvoice.frontend._extract_speech_token(prompt_speech_16k)
    # 将音色embedding、语音特征和语音token保存到字典中
    spk2info[speaker] = {'embedding': embedding, 'speech_feat': speech_feat, 'speech_token': speech_token}
    # 保存音色embedding
    torch.save(spk2info, spk2info_path)
print('Load time:', time.time() - start)
