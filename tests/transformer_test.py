import sys
sys.path.append("..")
from preprocess import PreprocessingPipeline
from train import train
from model import MusicTransformer
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
    seq_length = 128
    padded_length = 128
    pipeline = PreprocessingPipeline(input_dir=input_dir, stretch_factors=[0.975, 1, 1.025],
            split_size=30, sampling_rate=sampling_rate, n_velocity_bins=n_velocity_bins,
            transpositions=range(-2,3), training_val_split=training_val_split, max_encoded_length=seq_length+1, min_encoded_length=33)

    pipeline.run()
    training_sequences = pipeline.encoded_sequences['training']
    validation_sequences = pipeline.encoded_sequences['validation']
    n_states = 256 + sampling_rate + n_velocity_bins
    batch_size = 20
    optim="adam"
    transformer = MusicTransformer(n_states)
    train(transformer, training_sequences, validation_sequences, epochs = 2, padded_length=padded_length,
            evaluate_per=1, batch_size=batch_size, batches_per_print=1)


if __name__== "__main__":
    main()
