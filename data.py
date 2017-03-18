import os
import numpy as np
from itertools import islice

_sos = '<sos>'
_eos = '<eos>'

class Parser:
    def __init__(self, root, in_syms='isymbols', out_syms='osymbols'):
        with open(os.path.join(root, in_syms)) as isyms:
            self.in_syms = dict([(sym.strip(), i) for i, sym in enumerate(isyms)])
        with open(os.path.join(root, out_syms)) as osyms:
            self.out_syms = dict([(sym.strip(), i) for i, sym in enumerate(osyms)])

        self.out_syms_inv = {v: k for k, v in self.out_syms.items()}

    @property
    def input_size(self):
        return len(self.in_syms)

    @property
    def output_size(self):
        return len(self.out_syms)

    def parse_input(self, input_):
        input_ = [_sos] + list(input_.replace(' ', '_').lower()) + [_eos]
        return np.array([self.in_syms[tok] for tok in input_], dtype=np.int32)

    def parse_output(self, output_):
        # the output does not need _sos; but the symbol needs to be defined
        output_ = output_.split() + [_eos]
        return np.array([self.out_syms[tok] for tok in output_], dtype=np.int32)

    def compose_output(self, output_ids):
        return ' '.join([self.out_syms_inv[id] for id in output_ids])


class Reader:
    def __init__(self, root, data='data',
            in_syms='isymbols', out_syms='osymbols',
            in_maxlen=64, out_maxlen=64,
            batch_size=100):
        self.data_handle = open(os.path.join(root, data))
        self.parser = Parser(root, in_syms=in_syms, out_syms=out_syms)
        self.in_maxlen = in_maxlen
        self.out_maxlen = out_maxlen
        self.batch_size = batch_size

    @property
    def input_size(self):
        return self.parser.input_size

    @property
    def output_size(self):
        return self.parser.output_size

    def _input_ids(self, input_):
        ids = self.parser.parse_input(input_)
        if len(ids) > self.in_maxlen: # warn?
            ids = ids[:self.in_maxlen]
        pad = np.zeros([self.in_maxlen - len(ids)], np.int32) + self.parser.in_syms[_eos]
        ids = np.concatenate([ids, pad])
        return ids, len(ids)

    def _output_ids(self, output_):
        ids = self.parser.parse_output(output_)
        if len(ids) > self.out_maxlen: # warn?
            ids = ids[:self.out_maxlen]
        pad = np.zeros([self.out_maxlen - len(ids)], np.int32) + self.parser.out_syms[_eos]
        ids = np.concatenate([ids, pad])
        return ids, len(ids)

    def next_batch(self):
        data = [line.strip().split(' :: ') for line in islice(self.data_handle, self.batch_size)]
        if data and len(data) == self.batch_size:
            inputs, outputs = zip(*data)
            inputs_ids, inputs_len = zip(*[self._input_ids(i) for i in inputs])
            outputs_ids, outputs_len = zip(*[self._output_ids(o) for o in outputs])
            return (np.array(inputs_ids),
                    np.array(inputs_len),
                    np.array(outputs_ids),
                    np.array(outputs_len))
        else:
            return None, None, None, None
