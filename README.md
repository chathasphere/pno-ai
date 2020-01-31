# pno-ai
Pianistic Neuralnetworks Output Aleatoric Improvisations

### About
An implementation of Google Magenta's [Music Transformer](https://magenta.tensorflow.org/music-transformer) in Python/Pytorch. This library is designed to train a neural network on Piano MIDI data to generate musical samples. MIDIs are encoded into "Event Sequences", a dense array of musical instructions (note on, note off, dynamic change, time shift) encoded as numerical tokens. A custom transformer model learns to predict instructions on training sequences, and in `generate.py` a trained model can randomly sample from its learned distribution. (It is recommended to 'prime' the model's internal state with a MIDI input.)

### Training Data:
The initial dataset comes from several years of recordings from the International Piano-e-Competition: over 1,000 performances played by professional pianists on a Yamaha Disklavier. Obtainable [here](https://magenta.tensorflow.org/datasets/maestro). A sufficiently large dataset (order of 50 MB) of piano MIDIs should be sufficient to train a model. 

### Bibliography:
- Effective encoding of MIDI data: https://arxiv.org/abs/1808.03715
- Music Transformer AI Model: https://arxiv.org/abs/1809.04281
- Relative self-attention: https://arxiv.org/abs/1803.02155
- Original Transformer paper: https://arxiv.org/abs/1803.02155
- Guide to the above, with code: https://nlp.seas.harvard.edu/2018/04/03/attention.html
- Very readable introduction to attention/transformers: http://www.peterbloem.nl/blog/transformers


