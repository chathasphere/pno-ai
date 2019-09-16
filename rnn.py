import argparse
from preprocess import PreprocessingPipeline
from train import train
from model import MusicRNN
from helpers import sample

def main():
    parser = argparse.ArgumentParser("Script to train a music-generating RNN on piano MIDI data")
    parser.add_argument("--test", action='store_true',
            help="if true, use a smaller data set")
    parser.add_argument("--pack_batches", action='store_true',
            help="if true, pack batches with sequences of variable length, else just pad.")
    parser.add_argument("--batch_size", type=int, default=20,
            help="number of sequences per batch")
    parser.add_argument("--lr", type=float, default=.1,
            help="initializes model's learning rate")
    parser.add_argument("--momentum", type=float, default=.9,
            help="Nestorov momentum")
    parser.add_argument("--rnn_layers", type=int, default=1,
            help="number of stacked LSTM layers in model")
    parser.add_argument("--optim", type=str, default="sgd",
            choices=["sgd", "adam"], help="choice of optimizer")

    args = parser.parse_args()

    if args.test:
        print("Using smaller testing dataset...")
        input_dir = "data/test" 
        training_val_split = 0.7
    else:
        input_dir = "data/maestro-v2.0.0"
        training_val_split = 0.9

    #defines, in Hz, the smallest timestep preserved in quantizing MIDIs
    #determines number of timeshift events
    sampling_rate = 125
    #determines number of velocity events
    n_velocity_bins = 32
    pipeline = PreprocessingPipeline(input_dir=input_dir, stretch_factors=[0.975, 1, 1.025],
            split_size=15, sampling_rate=sampling_rate, n_velocity_bins=n_velocity_bins,
            transpositions=range(-2,3), training_val_split=training_val_split, max_encoded_length=513, min_encoded_length=33)


    pipeline.run()
    training_sequences = pipeline.encoded_sequences['training']
    validation_sequences = pipeline.encoded_sequences['validation']
    n_states = 256 + sampling_rate + n_velocity_bins
    hidden_size = 512
    pack_batches = args.pack_batches
    rnn = MusicRNN(n_states, hidden_size, batch_first = not(pack_batches))
    train(rnn, training_sequences, validation_sequences, epochs = 2,  
            lr=args.lr, evaluate_per=1, batch_size=args.batch_size, pack_batches=pack_batches)
    sample(rnn, sample_length=10)

if __name__ == "__main__":
    main()
