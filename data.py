import os
import numpy as np
from itertools import islice

class Reader:
    def __init__(self, root_path, data='data',
            isymbols='isymbols', osymbols='osymbols',
            imax_len=64, omax_len=64,
            batch_size=100):
        self.data_handle = open(os.path.join(root_path, data))
        with open(os.path.join(root_path, isymbols)) as isyms:
            self.isymbols = dict([(sym.strip(), i) for i, sym in enumerate(isyms)])
        with open(os.path.join(root_path, osymbols)) as osyms:
            self.osymbols = dict([(sym.strip(), i) for i, sym in enumerate(osyms)])
        self.imax_len = imax_len
        self.omax_len = omax_len
        self.batch_size = batch_size

    def _input_ids(self, input_):
        input_ = ['<s>'] + list(input_.replace(' ', '_').lower()) + ['</s>']
        if len(input_) > self.imax_len:
            raise Exception("input length exceeded imax_len")
        ids = np.array([self.isymbols[tok] for tok in input_], dtype=np.int32)
        ids = np.concatenate((ids, np.zeros([self.imax_len - len(input_)], np.int32) - 1))
        return ids, len(input_)

    def _output_ids(self, output_):
        output_ = ['<s>'] + output_.split() + ['</s>']
        if len(output_) > self.omax_len:
            raise Exception("output length exceeded omax_len")
        ids = np.array([self.osymbols[tok] for tok in output_], dtype=np.int32)
        ids = np.concatenate((ids, np.zeros([self.omax_len - len(output_)], np.int32) - 1))
        return ids, len(output_)

    def next_batch(self):
        inputs, outputs = zip(*[line.strip().split(' :: ')
                                for line in islice(self.data_handle, self.batch_size)])
        inputs_ids, inputs_len = zip(*[self._input_ids(i) for i in inputs])
        outputs_ids, outputs_len = zip(*[self._output_ids(o) for o in outputs])
        return (np.array(inputs_ids),
                np.array(inputs_len),
                np.array(outputs_ids),
                np.array(outputs_len))
