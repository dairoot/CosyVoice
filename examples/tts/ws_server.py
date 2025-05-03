import os
import sys

import torchaudio
import websockets

sys.path.append('./')
sys.path.append('./third_party/Matcha-TTS')
current_dir = os.path.dirname(os.path.abspath(__file__))
import time
from websockets.sync.server import serve

import json
import torch

from cosyvoice.cli.cosyvoice import CosyVoice2

with open('./examples/tts/audio/speaker_data.json', 'r', encoding='utf-8') as file:
    speaker_dict = json.load(file)

# 将第三方库Matcha-TTS的路径添加到系统路径中
sys.path.append('third_party/Matcha-TTS')

# 记录开始时间
start = time.time()
# 初始化CosyVoice2模型，指定预训练模型路径，不加载jit和trt模型，使用fp32

cosyvoice = CosyVoice2('iic/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=True)

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
def tts_sft(tts_text, stream=False, speed=1.0):
    print("tts_text", tts_text)
    speaker_info = spk2info[speaker]
    # 提取文本的token和长度
    tts_text_token, tts_text_token_len = cosyvoice.frontend._extract_text_token(tts_text)

    # 提取提示文本的token和长度
    prompt_text_token, prompt_text_token_len = cosyvoice.frontend._extract_text_token(
        speaker_dict[speaker]["prompt_text"])

    # 获取说话人的语音token长度，并转换为torch张量，移动到指定设备
    speech_token_len = torch.tensor(
        [speaker_info['speech_token'].shape[1]], dtype=torch.int32
    ).to(cosyvoice.frontend.device)

    # 获取说话人的语音特征长度，并转换为torch张量，移动到指定设备
    speech_feat_len = torch.tensor(
        [speaker_info['speech_feat'].shape[1]], dtype=torch.int32
    ).to(cosyvoice.frontend.device)

    # 构建模型输入字典，包括文本、文本长度、提示文本、提示文本长度、LLM提示语音token、LLM提示语音token长度、流提示语音token、流提示语音token长度、提示语音特征、提示语音特征长度、LLM嵌入和流嵌入
    model_input = {
        'text': tts_text_token,
        'text_len': tts_text_token_len,
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


def text_generator():
    yield '收到好友从远'
    yield '方寄来的礼'
    yield '物，那份意'
    yield '外的惊喜与深深的祝福'


def test_tts():
    # 初始化一个列表来收集音频块
    audio_chunks = []
    for j in tts_sft(text_generator(), stream=True, speed=1.2):
        audio_chunks.append(j['tts_speech'])

    # 将音频块合并为一个完整的音频文件
    torchaudio.save('./output.wav', torch.cat(audio_chunks, dim=1), sample_rate=cosyvoice.sample_rate)


# 定义一个函数，用于获取websocket消息
def get_ws_message(websocket):
    # 遍历websocket消息
    for message in websocket:
        print(f"Received: {message}", type(message))
        if message == "END_OF_AUDIO":
            break
        yield message


def echo(websocket):
    for j in tts_sft(get_ws_message(websocket), stream=True, speed=1.2):
        audio_bytes = j['tts_speech'].numpy().tobytes()
        websocket.send(audio_bytes)

    websocket.send(b"END_OF_AUDIO")


def main():
    with serve(echo, "0.0.0.0", 8765) as server:
        print("WebSocket server started on ws://0.0.0.0:8765")
        server.serve_forever()


if __name__ == "__main__":
    main()
    # test_tts()
