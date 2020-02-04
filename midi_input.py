import rtmidi
from pretty_midi import Note, PrettyMIDI, Instrument
from helpers import vectorize

class MidiInputError(Exception):
    pass

def read(n_velocity_events=32, n_time_shift_events=125):
    if True:
        midiin = rtmidi.MidiIn()
        available_ports = midiin.get_ports()
    
        if available_ports:
            print("Connecting to midi-in port!")
            midiin.open_port(0)
        else:
            raise MidiInputError("Midi ports not availabled...")
    
        msg_sequence = []
    
        while True:
            proceed = input("Play something on the keyboard and enter 'c' to continue or 'q' to quit.\n")
            if proceed == "c":
                midiin.close_port()
                break
            elif proceed == "q":
                return
            else:
                print("Command not recognized")
                continue
    
        while True:
            msg = midiin.get_message()
            if msg is None:
                break
            else:
                msg_sequence.append(msg)
    
    
        if len(msg_sequence) == 0:
            raise MidiInputError("No messages detected")

        note_sequence = []
        i = 0
        #notes that haven't ended yet
        live_notes = {}
        while i < len(msg_sequence):
            info , time_delta = msg_sequence[i]
            if i == 0:
                #start time tracking from zero
                time = 0
            else:
                #shift forward
                time = time + time_delta
            pitch = info[1]
            velocity = info[2]
            if velocity > 0:
                #(pitch (on), velocity, start_time (relative)
                live_notes.update({pitch: (velocity, time)})
                #how to preserve info ...?
            else:
                note_info = live_notes.get(pitch)
                if note_info is None:
                    raise MidiInputError("what?")
                note_sequence.append(Note(pitch=pitch, velocity = note_info[0], 
                    start = note_info[1], end = time))
                live_notes.pop(pitch)

            i += 1

        note_sequence = quantize(note_sequence, n_velocity_events, n_time_shift_events)

        note_sequence = vectorize(note_sequence)
        return note_sequence

def quantize(note_sequence, n_velocity_events, n_time_shift_events):

    timestep = 1 / n_time_shift_events
    velocity_step = 128 // n_velocity_events

    for note in note_sequence:
        note.start = (note.start * n_time_shift_events) // 1 * timestep
        note.end = (note.end * n_time_shift_events) // 1 * timestep

        note.velocity = (note.velocity // velocity_step) * velocity_step + 1


    return note_sequence

if __name__ == "__main__":
    read()



