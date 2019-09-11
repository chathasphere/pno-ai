from helpers import prepare_batches, one_hot
import torch
import torch.nn as nn
import time
from random import shuffle
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


def pad_batch(input_sequences, target_sequences, n_states, pack_batches):

    sequence_lengths = [len(s) for s in input_sequences]

    input_batch = pad_sequence([one_hot(s, n_states) for s in input_sequences], 
            batch_first=not(pack_batches))
    target_batch = pad_sequence([torch.tensor(s) for s in target_sequences],
            batch_first=not(pack_batches))

    if pack_batches:
        x = pack_padded_sequence(input_batch, sequence_lengths)
        y = pack_padded_sequence(target_batch, sequence_lengths)[0]

    else:
        x = input_batch
        y = target_batch.flatten()

    return x, y

def train(model, training_data, validation_data,
        epochs, lr, evaluate_per, batch_size, pack_batches):

    training_start_time = time.time()

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr) 
    loss_function = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        print("GPU is available")
    else:
        print("GPU not available, CPU used")

    for e in range(epochs):
        batch_start_time = time.time()
        batch_num = 1
        training_batches = prepare_batches(training_data, batch_size) #returning batches of a given size
        # input and target sequences, latter one time step ahead
        #potentially need some if/else logic to handle hidden states
        hx = None #this is the hidden state, None means: initializes to zeros
        # hidden state is the "internal representation" of the sequence

        for input_sequences, target_sequences in training_batches:

            #skip batches that are undersized
            if len(input_sequences) != batch_size:
                continue

            x, y = pad_batch(input_sequences, target_sequences, model.n_states, pack_batches)

            y_hat, hx = model(x, hx, pack_batches)

            loss = loss_function(y_hat, y)

            #detach hidden state from the computation graph; we don't need its gradient
            hx = tuple(h.detach() for h in hx)
            #clear old gradients from previous step
            model.zero_grad()
            #compute derivative of loss w/r/t parameters
            loss.backward()
            #optimizer takes a step based on gradient
            optimizer.step()
            training_loss = loss.item()
            #print(f"batch {batch_num}, loss: {training_loss : .2f}")
            batch_num += 1

        print(f"epoch: {e+1}/{epochs} | time: {time.time() - batch_start_time:.0f}s")
        print(f"training loss: {training_loss :.2f}")
        shuffle(training_data)

        if (e + 1) % evaluate_per == 0:

            #deactivate backprop for evaluation
            model.eval()
            validation_batches = prepare_batches(validation_data,
                    batch_size)
            #get loss per batch
            val_loss = 0
            n_batches = 0
            for input_sequences, target_sequences in validation_batches:

                if len(input_sequences) != batch_size:
                    continue

                x, y = pad_batch(input_sequences, target_sequences, model.n_states, pack_batches)

                y_hat, hx = model(x, hx, pack_batches)
                loss = loss_function(y_hat, y)
                val_loss += loss.item()
                n_batches += 1

            model.train()
            print(f"validation loss: {val_loss / n_batches:.2f}")
            shuffle(validation_data)

