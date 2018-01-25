# Text2phones using Tensorflow

A sequence-to-sequence neural network to generate phonetic transcriptions from
plain text.

The text is one-hot encoded and is passed through a 1-D convolution layer. The
output sequence from the convolution layer serves as memory units for
attentional decoding. The decoder has 5 RNN layers, 2 of which are used to
drive [`LuongMonotonicAttention`][] mechanism.

[`LuongMonotonicAttention`]: https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/LuongMonotonicAttention

## Training Data

I used [CMUDict][] to generate phonetic transcriptions of sentences for
training. Sentences were taken from [VCTK-Corpus][] speech transcriptions.

[CMUDict]: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
[VCTK-Corpus]: https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html

[A sample data file](./data-sample) is included with this repo which may be
used for training. Each line has the following format:

```
The end result is the same. :: DH AH _ EH N D _ R IH Z AH L T _ IH Z _ DH AH _ S EY M .
```

Double-colon (` :: `) splits plaintext from phonetic transcription. Plaintext
must only contain symbols listed in [`symbols/input`][], and the phonetic
transcription must only contain symbols listed in [`symbols/output`][],
separated by white spaces. `<eos>` and `<sos>` listed in symbol files indicate
end-of-sequence and start-of-sequence respectively, which are just placeholders
that will be inserted when data is loaded for training by [`data.py`][].

[`symbols/input`]: ./symbols/input
[`symbols/output`]: ./symbols/output
[`data.py`]: ./data.py

The data file I used for training contained the entire CMUDict as well as most
sentences from VCTK Corpus, and shuffled randomly. Download it
[here](https://gist.github.com/johncf/90f7a71d96e6d51d8dfd93ee3bb8e89a).

## Usage

To train the model, first verify the parameters provided at the beginning of
`train.py`, and run the following:

```
$ ./train.py
```

This will run the training procedure indefinitely. While training is in
progress, you may run `tensorboard --logdir=logdir/` to see the accuracy and
loss over time.

After enough training, run `./infer.py` and a prompt will await your input.

## Results

After nearly 2 hours of training (in my i5-6200U laptop without CUDA), the
training accuracy was around 45% for an output max-length of 16. Even though it
sounds pretty bad, the output looked pretty nice:

```text
in> hello world
HH EH L OW _ W AO L D <eos>

Attention alignment:
[ 1  2  3  5  6  7  8 10 11 12]

in> i felt like i could be the iron man if i put my neurons to it.
AY _ F EH L T _ L IH K _ AY _ K UW D _ B IY _ T AH _ IH R AH N _ M AE N _ IH F _ AY _ P AH T _ M AY _ N ER R AH Z S _ T UW _ IH T . <eos>

Attention alignment:
[ 1  2  3  4  5  6  7  8  9 10 12 13 14 15 17  ...  56 57 58 59 60 61 62 63]

in> never-seen-before words such as malayalam causes little hiccups.
N EH V ER S EH N B EH F ER R _ W ER D S _ S AH K _ AE Z _ M AE L EY AH _ _ K AH AH _ L IH T IH AH _ _ IH K AH AH AH S K S . . <eos>

Attention alignment:
[ 1  2  3  4  7  8 10 12 13  ...  34 35 36 41 42 42 44 45 47 49 50 51 51 51
 55 56 57 58 58 58 62 58 55 64 64 64 64 64]

in> without those, the hiccups are gone no matter how long the sentence is.
W IH T OW T _ T AH S _ DH AH _ HH IH K AH P S _ AA R _ G OW N _ N OW _ M AE T ER _ HH UW _ L AA NG _ T AH _ S EH N T AH N K _ IH S . <eos>

Attention alignment:
[ 1  2  3  5  7  8  9 11 12 14 16 18 19 20 21 22  ...  65 66 67 69 70 71 71]
```

## Miscellaneous

Old presentation slides describing my experience learning Tensorflow [at Google Docs][].

[at Google Docs]: https://docs.google.com/presentation/d/1WhFAsk6Cx7p0iJyAuZIu_7CTgl74jWPxSpntYN3IIw0/edit?usp=sharing
