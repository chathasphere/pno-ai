import sys
sys.path.append("..")
from preprocess import PreprocessingPipeline
from train import train
from model import MusicTransformer
from helpers import sample

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
    training_sequences = pipeline.encoded_sequences['training'][:1000]
    validation_sequences = pipeline.encoded_sequences['validation'][:100]
    n_tokens = 256 + sampling_rate + n_velocity_bins
    batch_size = 10
    optim="adam"
    transformer = MusicTransformer(n_tokens, seq_length=padded_length, d_model=4,
            d_feedforward=32, n_heads=4, positional_encoding=True,
            relative_pos=True)

    train(transformer, training_sequences, validation_sequences, epochs = 2, evaluate_per=1, batch_size=batch_size, batches_per_print=20,
            padding_index=0, checkpoint_path = "../saved_models/test_save", custom_schedule=True)

    print(sample(transformer, 10))


if __name__== "__main__":
    main()
