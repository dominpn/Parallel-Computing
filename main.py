from PyQt5 import QtCore, QtGui

from spectogram import Spectrogram, CHUNKS, RATE
from recorder import Recorder


def main():
    app = QtGui.QApplication([])
    spectrogram = Spectrogram()
    spectrogram.read_collected.connect(spectrogram.update_graph)

    mic = Recorder(spectrogram.read_collected)

    interval = RATE / CHUNKS
    timer = QtCore.QTimer()
    timer.timeout.connect(mic.read_from_stream)
    timer.start(1000 / interval)

    app.exec_()
    mic.close_stream()


if __name__ == '__main__':
    main()
