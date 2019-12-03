import sys
sys.path.append("..")

from preprocess import PreprocessingPipeline
from train import train
from model import MusicRNN
from generate import sample

def main():

    input_dir = "../data/test"
    training_val_split = 0.7
    #defines, in Hz, the smallest timestep preserved in quantizing MIDIs
    #determines number of timeshift events
    sampling_rate = 125
    #determines number of velocity events
    n_velocity_bins = 32


    #set up data pipeline
    pipeline = PreprocessingPipeline(input_dir=input_dir, stretch_factors=[0.975, 1, 1.025],
            split_size=15, sampling_rate=sampling_rate, n_velocity_bins=n_velocity_bins,
            transpositions=range(-2,3), training_val_split=training_val_split, max_encoded_length=513, min_encoded_length=33)

    pipeline.run()
    training_sequences = pipeline.encoded_sequences['training']
    validation_sequences = pipeline.encoded_sequences['validation']
    n_states = 256 + sampling_rate + n_velocity_bins
    hidden_size = 512
    pack_batches = True
    batch_size = 20
    rnn = MusicRNN(n_states, hidden_size, batch_first = not(pack_batches))
    train(rnn, training_sequences, validation_sequences, epochs = 2, 
            evaluate_per=1, batch_size=batch_size,
            pack_batches=pack_batches, batches_per_print=1)
    sample(rnn, sample_length=10)

if __name__ == "__main__":
    main()
