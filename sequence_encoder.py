from pretty_midi import Note

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

    def __init__(self, n_time_shift_events, n_velocity_events,
            sequences_per_update=50000, min_events=33, max_events=513):
        self.n_time_shift_events = n_time_shift_events
        self.n_events = 256 + n_time_shift_events + n_velocity_events
        self.timestep = 1 / n_time_shift_events
        self.velocity_bin_size = 128 // n_velocity_events
        self.sequences_per_update = sequences_per_update
        self.min_events = min_events
        self.max_events = max_events

    def encode_sequences(self, sample_sequences):
        """
        Converts each sample note sequence into an "event" sequence, a list of integers
        0 through N-1 where N is the total number of events in the encoder's
        representation.
        """
        event_sequences = []
        #count how many sequences are discarded/truncated due to length
        short_count, long_count = 0,0
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
                #extract start/end time, pitch and velocity
                t0, t1, p, v = note
                event_timestamps.append((t0, "VELOCITY", v))
                #if v != current_velocity:
                #    event_timestamps.append((t0, "VELOCITY", v))
                #    current_velocity = v
                event_timestamps.append((t0, "NOTE_ON", p))
                event_timestamps.append((t1, "NOTE_OFF", p))

            # sort events by timestamp
            event_timestamps = sorted(event_timestamps, key = lambda x: x[0])
            current_time = 0
            max_timeshift = self.n_time_shift_events
            #this loop encodes timeshifts as numbers
            #consider turning this into a function to help readability
            for timestamp in event_timestamps:
                #capture a shift in absolute time
                if timestamp[0] != current_time:
                    #convert to relative time and convert to number of quantized timesteps
                    timeshift = (timestamp[0] *  self.n_time_shift_events) - \
                            (current_time * self.n_time_shift_events)
                    #this is hacky but sue me
                    timeshift = int(timeshift + .1)
                    timeshift_events = []
                    #aggregate pauses longer than one second, as necessary
                    while timeshift > max_timeshift:
                        timeshift_events.append(
                                self.event_to_number("TIME_SHIFT", max_timeshift))
                        timeshift -= max_timeshift
                    #add timeshift (mod 1 second) as an event
                    timeshift_events.append(
                            self.event_to_number("TIME_SHIFT", timeshift))
                    event_sequence.extend(timeshift_events)
                    
                    #add the other events: NOTE_ON, NOTE_OFF, VELOCITY
                    current_time = timestamp[0]
                event_sequence.append(
                        self.event_to_number(timestamp[1], timestamp[2]))

            #check if sequence is too short to keep
            if self.min_events is not None:
                if len(event_sequence) < self.min_events:
                    short_count += 1
                    continue
            #truncate sequence if necessary
            if self.max_events is not None:
                if len(event_sequence) > self.max_events:
                    event_sequence = event_sequence[:self.max_events]
                    long_count += 1

            event_sequences.append(event_sequence)

        if short_count > 0:
            print(f"{short_count} sequences discarded due to brevity")
        if long_count > 0:
            print(f"{long_count} sequences truncated due to excessive length.")

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
            #subtract one to fit to zero-index convention
            #i.e. the number 256 corresponds to the smallest possible timestep
            #which is non-zero...!
            return value + 256 - 1
        elif event == "VELOCITY":
            #convert to bins
            v_bin = (value - 1) // self.velocity_bin_size
            return v_bin + 256 + self.n_time_shift_events
        else:
            raise SequenceEncoderError("Event type {} not recognized".format(event))

    def number_to_event(self, number):
        number = int(number)
        if number < 0 or number >= self.n_events:
            raise SequenceEncoderError("Number {} out of range")

        if number < 128:
            event = "NOTE_ON", number
        elif 128 <= number < 256:
            event = "NOTE_OFF", number - 128
        elif 256 <= number < 256 + self.n_time_shift_events:
            event = "TIME_SHIFT", number + 1 - 256
        else:
            bin_number = number - 256 - self.n_time_shift_events
            event = "VELOCITY", (bin_number * self.velocity_bin_size) + 1
        return event

    def decode_sequences(self, encoded_sequences):
        """
        Given a list of encoded sequences, decode each of them and return a list of pretty_midi Note sequences.
        """
        note_sequences = []
        for encoded_sequence in encoded_sequences:
            note_sequences.append(self.decode_sequence(encoded_sequence))

        return note_sequences

    def decode_sequence(self, encoded_sequence, stuck_note_duration=None, keep_ghosts=False, verbose=False):
        """
        Takes in an encoded event sequence (sparse numerical representation) and transforms it back into a pretty_midi Note sequence. Randomly-generated encoded sequences, such as produced by the generation script, can have some unusual traits such as notes without a provided end time. Contains logic to handle these pathological notes.

        Args:
            encoded_sequence (list): List of events encoded as integers
            stuck_note_duration (int or None): if defined, for recovered notes missing an endtime, give them a fixed duration (as number of seconds held)
            keep_ghosts (bool): if true, when the decoding algorithm recovers notes with an end time preceding their start time, keep them by swapping start and end. If false, discard the "ghost" notes
            verbose (bool): If true, print results on how many stuck notes and ghost notes are detected.
        """
        events = []
        for num in encoded_sequence:
            events.append(self.number_to_event(num))
        #list of pseudonotes = {'start':x, 'pitch':something, 'velocity':something}
        notes = []
        #on the second pass, add in end time
        note_ons = []
        note_offs = []
        global_time = 0
        current_velocity = 0
        for event, value in events:
            #check event type
            if event == "TIME_SHIFT":
                global_time += 0.008 * value
                global_time = round(global_time, 5)

            elif event == "VELOCITY":
                current_velocity = value
            
            elif event == "NOTE_OFF":
                #eventually we'll sort this by timestamp and work thru
                note_offs.append({"pitch": value, "end": global_time})
            
            elif event == "NOTE_ON":
                #it's a NOTE_ON!
                #value is pitch 
                note_ons.append({"start": global_time, 
                    "pitch": value, "velocity": current_velocity})
            else:
                raise SequenceEncoderError("you fool!")

        #keep a count of notes that are missing an end time (stuck notes)
        #----default behavior is to ignore them. 
        stuck_notes = 0
        
        #keep a count of notes assigned end times *before* their start times (ghost notes)
        #----default behavior is to ignore them
        ghost_notes = 0


        #Zip up notes with corresponding note-off events
        while len(note_ons) > 0:
            note_on = note_ons[0]
            pitch = note_on['pitch']
            #this assumes everything is sorted nicely!
            note_off = next((n for n in note_offs if n['pitch'] == pitch), None)
            if note_off == None:
                stuck_notes += 1
                if stuck_note_duration is None:
                    note_ons.remove(note_on)
                    continue
                else:
                    note_off = {"pitch": pitch, "end": note_on['start'] + stuck_note_duration}
            else:
                note_offs.remove(note_off)

            if note_off['end'] < note_on['start']:
                ghost_notes += 1
                if keep_ghosts:
                    #reverse start and end (and see what happens...!)
                    new_end = note_on['start']
                    new_start = note_off['end']
                    note_on['start'] = new_start
                    note_off['end'] = new_end
                else:
                    note_ons.remove(note_on)
                    continue

            note = Note(start = note_on['start'], end = note_off['end'],
                    pitch = pitch, velocity = note_on['velocity'])
            notes.append(note)
            note_ons.remove(note_on)

        if verbose:
            print(f"{stuck_notes} notes missing an end-time...")
            print(f"{ghost_notes} had an end-time precede their start-time")

        return notes
            



