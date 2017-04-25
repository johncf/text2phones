# Tensorflow Attention Test

To run the model, first make a data file with each line of the following format:

```
The end result is the same. :: DH AH _ EH N D _ R IH Z AH L T _ IH Z _ DH AH _ S EY M .
```

Each line is split by ` :: `, every character in the first part is parsed using
[`isymbols`](./isymbols), and the second part is further split on white spaces
and parsed using [`osymbols`](./osymbols). A sample [`data`](./data) file is
included with this repository.

For a larger data file go [here](https://gist.github.com/johncf/90f7a71d96e6d51d8dfd93ee3bb8e89a).

To train the model, first verify the parameters provided at the beginning of
`train.py`, and run the following:

```
$ mkdir ckpts logdir
$ ./train.py
```

This will run the training procedure in infinite loop, while printing the
training progress. I haven't been able to get the accuracy significantly above
30% with the current setup.

While training is in progress, you may run `tensorboard --logdir=logdir/` and
see the accuracy and loss values over time.

Once the training is done to your satisfaction, run `./infer` and a prompt will
await your input. Enter a line of text and be amazed at its inaccuracy!!
