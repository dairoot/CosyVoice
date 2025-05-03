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
    ttt = """唧唧复唧唧，木兰当户织。
不闻机杼声，唯闻女叹息。
问女何所思？问女何所忆？
女亦无所思，女亦无所忆。
昨夜见军帖，可汗大点兵，
军书十二卷，卷卷有爷名。
阿爷无大儿，木兰无长兄，
愿为市鞍马，从此替爷征。

东市买骏马，西市买鞍鞯，
南市买辔头，北市买长鞭。
朝辞爷娘去，暮宿黄河边。
不闻爷娘唤女声，但闻黄河流水鸣溅溅。
旦辞黄河去，暮至黑山头。
不闻爷娘唤女声，但闻燕山胡骑声啾啾。

万里赴戎机，关山度若飞。
朔气传金柝，寒光照铁衣。
将军百战死，壮士十年归。

归来见天子，天子坐明堂。
策勋十二转，赏赐百千强。"""
    for line in ttt.split('\n'):
        for i in line.split("，"):
            for j in i.split("。"):
                yield j


start_time = time.time()
for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            text_generator(), speaker_dict[speaker]["prompt_text"], prompt_speech_16k, stream=False)):
    print(j['tts_speech'], cosyvoice)
    torchaudio.save(current_dir + '/audio/test_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

print("耗时：", time.time() - start_time)
