git submodule update --init --recursive

git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd

```bash
# 训练音色
python examples/tts/train_speaker.py

# 通过音色 生成 tts
python examples/tts/tts_by_speaker.py

# 通过音频 生成 tts
python examples/tts/tts_by_audio.py
```
 

