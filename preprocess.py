#!/usr/bin/env python3
import os
import pdb
from pretty_midi import PrettyMIDI, Instrument
import six
import copy

class PreprocessingError(Exception):
    pass

#consider implementing custom Note class:
#natural end time + altered end time

_SUSTAIN_ON = 0
_SUSTAIN_OFF = 1
_NOTE_ON = 2
_NOTE_OFF = 3

def convert_files():
    pretty_midis = []
    folders = [d for d in os.listdir(os.getcwd()) if os.path.isdir(d)]
    if len(folders) > 0:
        for d in folders:
            os.chdir(d)
            pretty_midis += convert_files()
            os.chdir("..")
    midis = [f for f in os.listdir(os.getcwd()) if \
            (f.endswith(".mid") or f.endswith("midi"))]
    for m in midis:
        with open(m, "rb") as f:
            try:
                midi_str = six.BytesIO(f.read())
                pretty_midis.append(PrettyMIDI(midi_str))
                print("Successfully parsed {}!".format(m))
            except:
                print("Could not parse {}".format(m))
    return pretty_midis


def apply_sustain(piano_data):
    """
    While the sustain pedal is applied during a midi, extend the length of all notes to the beginning of the next note of the same pitch or to the end of the sustain. Returns a 
    midi notes sequence.
    """
    notes = copy.deepcopy(piano_data.notes)
    control_changes = piano_data.control_changes
    #sequence of SUSTAIN_ON, SUSTAIN_OFF, NOTE_ON, and NOTE_OFF actions
    first_sustain_control = next(c for c in control_changes if c.number == 64)
    sustain_position = _SUSTAIN_ON if first_sustain_control.value >= 64 else _SUSTAIN_OFF
    action_sequence = [(first_sustain_control.time, sustain_position, None)]
    #delete this please
    cleaned_controls = []
    for c in control_changes:
        #Ignoring the sostenuto and damper pedals due to complications
        if sustain_position == _SUSTAIN_ON:
            if c.value >= 64:
                #another SUSTAIN_ON
                continue
            else:
                sustain_position = _SUSTAIN_OFF 
        else:
            #look for the next on signal
            if c.value < 64:
                #another SUSTAIN_OFF
                continue
            else:
                sustain_position = _SUSTAIN_ON 
        action_sequence.append((c.time, sustain_position, None))
        cleaned_controls.append((c.time, sustain_position))

    action_sequence.extend([(note.start, _NOTE_ON, note) for note in notes])
    action_sequence.extend([(note.end, _NOTE_OFF, note) for note in notes])
    #sort actions by time and type

    action_sequence = sorted(action_sequence, key = lambda x: (x[0], x[1]))
    live_notes = []
    sustain = False
    for action in action_sequence:
        if action[1] == _SUSTAIN_ON:
            sustain = True
        elif action[1] == _SUSTAIN_OFF:
            #find when the sustain pedal is released
            off_time = action[0]
            for note in live_notes:
                if note.end < off_time:
                    #shift the end of the note to when the pedal is released
                    note.end = off_time
                    live_notes.remove(note)
            sustain = False
        elif action[1] == _NOTE_ON:
            current_note = action[2]
            if sustain:
                for note in live_notes:
                    # if there are live notes of the same pitch being held, kill 'em
                    if current_note.pitch == note.pitch:
                        note.end = current_note.start
                        live_notes.remove(note)
            live_notes.append(current_note)
        else:
            if sustain == True:
                continue
            else:
                note = action[2]
                live_notes.remove(note)
    return notes

def write_to_midi(note_sequences, output_dir):
    i = 1
    for note_sequence in note_sequences:
        midi = PrettyMIDI(initial_tempo = 80)
        piano = Instrument(program=0, is_drum=False, name="test{}".format(i))
        piano.notes = note_sequence
        midi.instruments.append(piano)
        output_name = output_dir + "/test{}.midi".format(i)
        i += 1
        #with open(output_name, 'w')
        midi.write(output_name)

def main():
    #Perhaps this should be converted to a parsed CL arg
    home_dir = os.getcwd()
    input_dir = "data/test"
    output_dir = "output/test"
    os.chdir(input_dir)
    midis = convert_files()
    total_time = sum([m.get_end_time() for m in midis])
    print("{} midis read, or {:.1f} minutes of music".format(len(midis), total_time))

    note_sequences = []
    for m in midis:
        if m.instruments[0].program == 0:
            piano_data = m.instruments[0]
        else:
            #todo: write logic to safely catch if there are non piano instruments,
            #or extract the piano midi if it exists
            raise PreprocessingError("Non-piano midi detected")
        note_sequences.append(apply_sustain(piano_data))
    os.chdir(home_dir)
    write_to_midi(note_sequences, output_dir)    

if __name__ == "__main__":
    main()
