import pyaudio
import numpy as np

# settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNKS = 1024


class Recorder:
    def __init__(self, signal):
        self.signal = signal
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNKS)

    def read_from_stream(self):
        data = self.stream.read(CHUNKS, exception_on_overflow=False)
        y = np.fromstring(data, 'int16')
        self.signal.emit(y)

    def close_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
