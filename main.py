import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt

CHUNK = 1024 * 8
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


def main():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK
    )

    fig, ax = plt.subplots()

    x = np.arange(0, 2 * CHUNK, 2)
    line, = ax.plot(x, np.random.rand(CHUNK))
    ax.set_ylim(0, 255)
    ax.set_xlim(0, CHUNK)

    while True:
        data = stream.read(CHUNK)
        data_int = np.array(struct.unpack(str(2 * CHUNK) + 'B', data), dtype='b')[::2] + 128
        line.set_ydata(data_int)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.000000000000001)


if __name__ == '__main__':
    main()
