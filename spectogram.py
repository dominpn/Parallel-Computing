import numpy as np
from PyQt5 import QtCore
from pyqtgraph import PlotWidget, ColorMap, ImageItem
import requests

from recorder import CHUNKS, RATE


class Spectrogram(PlotWidget):
    read_collected = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(Spectrogram, self).__init__()

        self.img = ImageItem()
        self.addItem(self.img)

        self.img_array = np.zeros((1000, int(CHUNKS/2)+1))

        self.create_color_map()

        self.set_y_axis()

        self.win = np.hanning(CHUNKS)
        self.show()

    def create_color_map(self):
        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array(
            [
                [0, 255, 255, 255],
                [255, 255, 0, 255],
                [0, 0, 0, 255],
                (0, 0, 255, 255),
                (255, 0, 0, 255)
            ],
            dtype=np.ubyte)
        color_map = ColorMap(pos, color)
        lut = color_map.getLookupTable(0.0, 1.0, 256)

        self.img.setLookupTable(lut)
        self.img.setLevels([-50, 40])

    def set_y_axis(self):
        freq = np.arange((CHUNKS / 2) + 1) / (float(CHUNKS) / RATE)
        y_scale = 1.0 / (self.img_array.shape[1] / freq[-1])
        self.img.scale((1. / RATE) * CHUNKS, y_scale)

        self.setLabel('left', 'Frequency', units='Hz')

    def update_graph(self, chunks):
        res = requests.post('http://localhost:5000/', json={"chunks": np.array(chunks[:513]).tolist()})
        if res.ok:
            self.img_array = np.roll(self.img_array, -1, 0)
            self.img_array[-1:] = res.json()
            self.img.setImage(self.img_array, autoLevels=False)
