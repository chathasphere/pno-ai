from helpers import prepare_batches, sequences_to_tensor
import torch
import torch.nn as nn
import time
from random import shuffle
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def train(model, training_data, validation_data,
        epochs, lr, evaluate_per, batch_size, pack_sequences, n_states):

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr) 
    loss_function = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        print("GPU is available")
    else:
        print("GPU not available, CPU used")

    for e in range(epochs):
        start_time = time.time()
        training_batches = prepare_batches(training_data, batch_size) #returning batches of a given size
        # input and target sequences, latter one time step ahead
        #potentially need some if/else logic to handle hidden states
        hx = None #this is the hidden state, None means: initializes to zeros
        # hidden state is the "internal representation" of the sequence

        for input_sequences, target_sequences in training_batches:

            #skip batches that are undersized
            if len(input_sequences) != batch_size:
                continue

            x = sequences_to_tensor(input_sequences, is_input=True, pack_sequences)

            y_hat, hx = model(input_tensors, hx)

            y = sequences_to_tensor(target_sequences, is_input=False, pack_sequences) 

            pdb.set_trace()

            loss = loss_function(y_hat.flatten(0,1), y)

            #detach hidden state from the computation graph; we don't need its gradient
	    hx = tuple(h.detach() for h in hx)
            #clear old gradients from previous step
            model.zero_grad()
            #compute derivative of loss w/r/t parameters
            loss.backward()
            #optimizer takes a step based on gradient
            optimizer.step()
            training_loss = loss.item()

        print(f"epoch: {e+1}/{epochs} | time: {time.time() - start_time:.0f}s")
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

                x = sequences_to_tensor(input_sequences, is_input=True, pack_sequences)

                y_hat, hx = model(input_sequences, hx)

                y = sequences_to_tensor(target_sequences, is_input=False, pack_sequences)

                pdb.set_trace()

                loss = loss_function(y_hat.flatten(0,1), y)
                val_loss += loss.item()
                n_batches += 1

            model.train()
            print(f"validation loss: {val_loss / n_batches:.2f}")
            shuffle(validation_data)

