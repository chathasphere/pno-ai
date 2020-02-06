from helpers import prepare_batches
from .custom import Accuracy, smooth_cross_entropy, TFSchedule
import torch
import torch.nn as nn
import time
from random import shuffle

def batch_to_tensors(batch, n_tokens, max_length):
    """
    Make input, input mask, and target tensors for a batch of seqa batch of sequences.
    """
    input_sequences, target_sequences = batch
    sequence_lengths = [len(s) for s in input_sequences]
    batch_size = len(input_sequences)

    x = torch.zeros(batch_size, max_length, dtype=torch.long)
    #padding element
    y = torch.zeros(batch_size, max_length, dtype=torch.long)


    for i, sequence in enumerate(input_sequences):
        seq_length = sequence_lengths[i]
        #copy over input sequence data with zero-padding
        #cast to long to be embedded into model's hidden dimension
        x[i, :seq_length] = torch.Tensor(sequence).unsqueeze(0)
    
    x_mask = (x != 0)
    x_mask = x_mask.type(torch.uint8)

    for i, sequence in enumerate(target_sequences):
        seq_length = sequence_lengths[i]
        y[i, :seq_length] = torch.Tensor(sequence).unsqueeze(0)

    if torch.cuda.is_available():
        return x.cuda(), y.cuda(), x_mask.cuda()
    else:
        return x, y, x_mask 

def train(model, training_data, validation_data,
        epochs, batch_size, batches_per_print=100, evaluate_per=1,
        padding_index=-100, checkpoint_path=None,
        custom_schedule=False, custom_loss=False):
    """
    Training loop function.
    Args:
        model: MusicTransformer module
        training_data: List of encoded music sequences
        validation_data: List of encoded music sequences
        epochs: Number of iterations over training batches
        batch_size: _
        batches_per_print: How often to print training loss
        evaluate_per: calculate validation loss after this many epochs
        padding_index: ignore this sequence token in loss calculation
        checkpoint_path: (str or None) If defined, save the model's state dict to this file path after validation
        custom_schedule: (bool) If True, use a learning rate scheduler with a warmup ramp
        custom_loss: (bool) If True, set loss function as Cross Entropy with label smoothing
    """

    training_start_time = time.time()

    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    if custom_schedule:
        optimizer = TFSchedule(optimizer, model.d_model)
    
    if custom_loss:
        loss_function = smooth_cross_entropy
    else:
        loss_function = nn.CrossEntropyLoss(ignore_index=padding_index)
    accuracy = Accuracy()

    if torch.cuda.is_available():
        model.cuda()
        print("GPU is available")
    else:
        print("GPU not available, CPU used")

    training_losses = []
    validation_losses = []
    #pad to length of longest sequence
    #minus one because input/target sequences are shifted by one char
    max_length = max((len(L) 
        for L in (training_data + validation_data))) - 1
    for e in range(epochs):
        batch_start_time = time.time()
        batch_num = 1
        averaged_loss = 0
        averaged_accuracy = 0
        training_batches = prepare_batches(training_data, batch_size) #returning batches of a given size
        for batch in training_batches:

            #skip batches that are undersized
            if len(batch[0]) != batch_size:
                continue
            x, y, x_mask = batch_to_tensors(batch, model.n_tokens, 
                    max_length)
            y_hat = model(x, x_mask).transpose(1,2)

            #shape: (batch_size, n_tokens, seq_length)

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
            averaged_accuracy += accuracy(y_hat, y, x_mask)
            if batch_num % batches_per_print == 0:
                print(f"batch {batch_num}, loss: {averaged_loss / batches_per_print : .2f}")
                print(f"accuracy: {averaged_accuracy / batches_per_print : .2f}")
                averaged_loss = 0
                averaged_accuracy = 0
            batch_num += 1

        print(f"epoch: {e+1}/{epochs} | time: {(time.time() - batch_start_time) / 60:,.0f}m")
        shuffle(training_data)

        if (e + 1) % evaluate_per == 0:

            #deactivate backprop for evaluation
            model.eval()
            validation_batches = prepare_batches(validation_data,
                    batch_size)
            #get loss per batch
            val_loss = 0
            n_batches = 0
            val_accuracy = 0
            for batch in validation_batches:

                if len(batch[0]) != batch_size:
                    continue

                x, y, x_mask = batch_to_tensors(batch, model.n_tokens, 
                        max_length)

                y_hat = model(x, x_mask).transpose(1,2)
                loss = loss_function(y_hat, y)
                val_loss += loss.item()
                val_accuracy += accuracy(y_hat, y, mask)
                n_batches += 1

            if checkpoint_path is not None:
                try:
                    torch.save(model.state_dict(),
                            checkpoint_path+f"_e{e}")
                    print("Checkpoint saved!")
                except:
                    print("Error: checkpoint could not be saved...")

            model.train()
            #average out validation loss
            val_accuracy = (val_accuracy / n_batches)
            val_loss = (val_loss / n_batches)
            validation_losses.append(val_loss)
            print(f"validation loss: {val_loss:.2f}")
            print(f"validation accuracy: {val_accuracy:.2f}")
            shuffle(validation_data)

    return training_losses

