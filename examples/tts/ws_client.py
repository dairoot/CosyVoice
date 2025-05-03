import asyncio
import wave

import av
import numpy as np
import websockets

wf1 = wave.open("./output1.wav", 'wb')
wf1.setnchannels(1)  # Mono audio
wf1.setsampwidth(2)  # 16-bit
wf1.setframerate(24000)  # Sample rate

wf2 = wave.open("./output2.wav", 'wb')
wf2.setnchannels(2)  # Mono audio
wf2.setsampwidth(2)  # 16-bit
wf2.setframerate(48000)  # Sample rate

resampler = av.AudioResampler(format='s16', layout='stereo', rate=48000)


async def connect():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Send the message to server
        await websocket.send("你好啊，你是什么模型")
        await websocket.send("型，我需要你帮我检")
        await websocket.send("测一下音频中的语")
        await websocket.send("音活动")

        await websocket.send("END_OF_AUDIO")

        while True:
            audio_data = await websocket.recv()
            if audio_data == b"END_OF_AUDIO":
                break
            # Convert bytes to numpy array and scale to int16
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            audio_array = (audio_array * 32767).astype(np.int16)
            # print(audio_array.shape)
            # 保存 24000hz
            wf1.writeframes(audio_array.tobytes())

            # 保存 48000hz stereo
            audio_array = audio_array.reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(audio_array, format='s16', layout='mono')
            frame.rate = 24000
            resampled_frames = resampler.resample(frame)
            print(resampled_frames)
            resampled_array = resampled_frames[0].to_ndarray()
            wf2.writeframes(resampled_array.tobytes())


if __name__ == "__main__":
    asyncio.run(connect())
