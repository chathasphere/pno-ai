class SequenceEncoderError(Exception):
    pass

class SequenceEncoder():
    """
    Converts sequences of Midi Notes to sequences of events under the following 
    representation:
    - 128 NOTE-ON events (for each of the 128 MIDI pitches, starts a new note)
    - 128 NOTE-OFF events (likewise. Ends 
    - (1000 / t) TIME-SHIFT events (each moves the time step forward by increments of 
      t ms up to 1 second
    - v VELOCITY events (each one changes the velocity applied to all subsequent notes
      until another velocity event occurs)

    Includes functions to cast a sequence of Midi Notes to a numeric list of 
    possible events and to one-hot encode a numeric sequence as a Pytorch tensor.
    """
    action_ordering = {
            "VELOCITY": 0,
            "NOTE_OFF": 1,
            "NOTE_ON": 2
            }

    def __init__(self, n_time_shift_events, n_velocity_events,
            sequences_per_update=100):
        self.n_time_shift_events = n_time_shift_events
        self.n_velocity_events = n_velocity_events
        self.n_events = 256 + n_time_shift_events + n_velocity_events
        self.timestep = 1 / n_time_shift_events
        self.sequences_per_update = sequences_per_update

    def encode_event_sequences(self, sample_sequences):
        """
        Converts each sample note sequence into an "event" sequence, a list of integers
        0 through N-1 where N is the total number of events in the encoder's
        representation.
        """
        event_sequences = []
        n_sequences = len(sample_sequences)
        for i in range(n_sequences):
            if not (i % self.sequences_per_update):
                print("{:,} / {:,} sequences encoded".\
                        format(i, n_sequences))
            event_sequence = []
            event_timestamps = []
            #attempt at efficiency gain: only add a velocity event if it's different
            #from current velocity...this is tricky if two notes played at the
            #same time have different velocity
            #current_velocity = 0
            for note in sample_sequences[i]:
                t0 = note.start
                t1 = note.end
                v = note.velocity
                p = note.pitch
                event_timestamps.append((t0, "VELOCITY", v))
                #if v != current_velocity:
                #    event_timestamps.append((t0, "VELOCITY", v))
                #    current_velocity = v
                event_timestamps.append((t0, "NOTE_ON", p))
                event_timestamps.append((t1, "NOTE_OFF", p))

            # sort events by timestamp and action
            event_timestamps = sorted(event_timestamps, 
                    key = lambda x: (x[0], SequenceEncoder.action_ordering[x[1]]))

            current_time = 0
            max_timeshift = self.n_time_shift_events
            for timestamp in event_timestamps:
                #capture a timeshift
                if timestamp[0] != current_time:
                    #I feel cautious about this casting...
                    timeshift = int((timestamp[0] / self.timestep) - \
                            (current_time / self.timestep))
                    timeshift_events = []
                    #aggregate pauses longer than one second, as necessary
                    while timeshift > max_timeshift:
                        #timeshift_events.append(("TIME_SHIFT", 125))
                        timeshift_events.append(
                                self.event_to_number("TIME_SHIFT", max_timeshift))
                        timeshift -= max_timeshift
                    #add timeshift (mod 1 second) as an event
                    timeshift_events.append(
                            self.event_to_number("TIME_SHIFT", timeshift))
                    event_sequence.extend(timeshift_events)
                    
                    #add the other events: NOTE_ON, NOTE_OFF, VELOCITY
                    event_sequence.append(
                            self.event_to_number(timestamp[1], timestamp[2]))
                    current_time = timestamp[0]

            event_sequences.append(event_sequence)

        return event_sequences
                
    def event_to_number(self, event, value):
        """
        Encode an event/value pair as a number 0-N-1
        where N is the number of unique events in the Encoder's representation.
        """
        if event == "NOTE_ON":
            return value
        elif event == "NOTE_OFF":
            return value + 128
        elif event == "TIME_SHIFT":
            return value + 256
        elif event == "VELOCITY":
            return value + 256 + self.n_time_shift_events
        else:
            raise SequenceEncoderError("Event type {} not recognized".format(event))


