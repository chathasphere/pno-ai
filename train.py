from helpers import prepare_batches, one_hot
import torch
import torch.nn as nn
import time
from random import shuffle


def pad_batch(input_sequences, target_sequences, n_states,
        padded_length = 256):

    sequence_lengths = [len(s) for s in input_sequences]
    batch_size = len(input_sequences)

    x = torch.zeros(batch_size, padded_length, dtype=torch.long)
    #right shape??? i don't think so
    input_mask = torch.tensor(x)
    y = torch.zeros(batch_size, padded_length, dtype=torch.long)
    target_mask = torch.tensor(y)

    for i, sequence in enumerate(input_sequences):
        seq_length = sequence_lengths[i]
        #copy over input sequence data with zero-padding
        #cast to long to be embedded into model's hidden dimension
        x[i, :seq_length] = torch.Tensor(sequence).unsqueeze(0)
        ##input mask: 0 for padded entries, else 1
        #sequence_mask = sequence.sum(1).unsqueeze(1).repeat(1, n_states)
        #input_mask[i, :seq_length, :] = sequence_mask.unsqueeze(0)


    for i, sequence in enumerate(target_sequences):
        seq_length = sequence_lengths[i]
        y[i, :seq_length] = torch.Tensor(sequence).unsqueeze(0)

    return (x.cuda(), y.cuda()) if torch.cuda.is_available() else (x, y)

def train(model, training_data, validation_data,
        epochs, evaluate_per, batch_size, padded_length,
        batches_per_print=100):

    training_start_time = time.time()

    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        print("GPU is available")
    else:
        print("GPU not available, CPU used")

    #start a list of mini-batch training losses
    training_losses = []
    for e in range(epochs):
        batch_start_time = time.time()
        batch_num = 1
        averaged_loss = 0
        training_batches = prepare_batches(training_data, batch_size) #returning batches of a given size

        for input_sequences, target_sequences in training_batches:

            #skip batches that are undersized
            if len(input_sequences) != batch_size:
                continue

            x, y = pad_batch(input_sequences, target_sequences, model.n_states, padded_length)
            
            y_hat = model(x, y).transpose(1,2)
            #shape: (batch_size, n_states, seq_length)

            loss = loss_function(y_hat, y)

            #detach hidden state from the computation graph; we don't need its gradient
            #clear old gradients from previous step
            model.zero_grad()
            #compute derivative of loss w/r/t parameters
            loss.backward()
            #optimizer takes a step based on gradient
            optimizer.step()
            training_loss = loss.item()
            training_losses.append(training_loss)
            #take average over subset of batch?
            averaged_loss += training_loss
            if batch_num % batches_per_print == 0:
                print(f"batch {batch_num}, loss: {averaged_loss / batches_per_print : .2f}")
                averaged_loss = 0
            batch_num += 1

        print(f"epoch: {e+1}/{epochs} | time: {time.time() - batch_start_time:.0f}s")
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

                x, y = pad_batch(input_sequences, target_sequences, model.n_states, padded_length)

                y_hat = model(x,y).transpose(1,2)
                loss = loss_function(y_hat, y)
                val_loss += loss.item()
                n_batches += 1

            model.train()
            print(f"validation loss: {val_loss / n_batches:.2f}")
            shuffle(validation_data)

    return training_losses

