import whisper

import sounddevice as sd
import numpy as np
import pyttsx3


class Transcriber:
    def __init__(self, model_size="small", duration=6, samplerate=16000):
        self.__model = whisper.load_model(model_size)
        self.__samplerate = samplerate
        self.__duration = duration
        self.engine = pyttsx3.init()

    def record_audio(self, ):
        print("Recording...")
        audio = sd.rec(int(self.__samplerate * self.__duration), samplerate=self.__samplerate, channels=1, dtype='float32')
        sd.wait()
        print("Recording complete.")
        return np.squeeze(audio)

    def transcribe_audio(self, audio_array):
        audio = whisper.pad_or_trim(audio_array)
        mel = whisper.log_mel_spectrogram(audio).to(self.__model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(self.__model, mel, options)
        return result.text

    def record_and_transcribe(self):
        audio_array = self.record_audio()
        transcription = self.transcribe_audio(audio_array)
        return transcription

    def say(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


if __name__ == "__main__":
    transcriber = Transcriber()
    # transcr = transcriber.record_and_transcribe()
    transcr = "smth"
    transcriber.say(transcr)
