import itertools

from flask import Flask, jsonify, request
from mpi4py import MPI
import numpy as np
from scipy.fftpack import dct

from recorder import CHUNKS

comm = MPI.COMM_WORLD

app = Flask(__name__)


@app.route("/", methods=['POST'])
def home():
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        data = request.json
        if 'chunks' in data:
            test_chunks = np.array_split(np.array(data['chunks']), size, axis=0)
        else:
            test_chunks = None

    else:
        test_chunks = None

    data = comm.scatter(test_chunks, root=0)

    print(f'Process {rank} received {data}')
    spec = abs(dct(data, 2) / CHUNKS)
    psd = 20 * np.log10(spec)

    processed_signal = comm.gather(psd, root=0)
    if rank == 0:
        return jsonify(list(itertools.chain(*processed_signal)))


if __name__ == '__main__':
    if comm.Get_rank() == 0:
        app.run(host='0.0.0.0', debug=True, port=5000+comm.Get_rank())
