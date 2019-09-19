# pno-ai
Pianistic Neuralnets Offer Aleatoric Improvisations

### About
A deep learning project written in Python & Pytorch to generate piano MIDI tracks.<br>
Transform Piano MIDI data into "Event Sequences," a sparse MIDI representation giving instructions to play notes, release them, change dynamics, and shift time. This encoding is an efficient, computer-readable format for expressive piano music. After training sequence-to-sequence neural networks (such as RNNs or Transformers) on a preprocessed dataset, you can generate random samples of "learned" music with the `generate.py` file.

Inspired by [this](https://magenta.tensorflow.org/music-transformer) blog post from Magenta.

### Training Data:
The initial dataset comes from several years of recordings from the International Piano-e-Competition: over 1,000 performances played by professional pianists on a Yamaha Disklavier. Obtainable [here](https://magenta.tensorflow.org/datasets/maestro). A sufficiently large dataset (order of 50 MB) of piano MIDIs should be sufficient to train a model. 

### Bibliography:
- Effective encoding of MIDI data: https://arxiv.org/abs/1808.03715
- Music Transformer AI Model: https://arxiv.org/abs/1809.04281
- Relative self-attention: https://arxiv.org/abs/1803.02155
- Original Transformer paper: https://arxiv.org/abs/1803.02155
- Seq2Seq architecture: https://arxiv.org/abs/1409.3215


