import sys
import os
import json

import time

sys.path.append('./')
sys.path.append('./third_party/Matcha-TTS')

current_dir = os.path.dirname(os.path.abspath(__file__))
# pip install matcha-tts


from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio  # type: ignore

with open('./examples/tts/audio/speaker_data.json', 'r', encoding='utf-8') as file:
    speaker_dict = json.load(file)

print(speaker_dict)
speaker = "leijun"

model_path = 'iic/CosyVoice2-0.5B'
cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
prompt_speech_16k = load_wav(speaker_dict[speaker]["audio"], 16000)


def text_generator():
    # yield '收到好友从远方寄来的生日礼物，'
    # yield '那份意外的惊喜与深深的祝福'
    # yield '让我心中充满了甜蜜的快乐，'
    # yield '笑容如花儿般绽放。'

    tts_text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    return tts_text


start_time = time.time()
for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            text_generator(), speaker_dict[speaker]["prompt_text"], prompt_speech_16k, stream=False)):
    print(j['tts_speech'], cosyvoice)
    torchaudio.save(current_dir + '/audio/test_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

print("耗时：", time.time() - start_time)
