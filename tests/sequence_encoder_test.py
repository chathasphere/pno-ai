from pretty_midi import Note
import sys, pdb
sys.path.append("..")
from preprocess import SequenceEncoder
from helpers import vectorize

sample_note_sequence0 = [
    [Note(start=0.928000, end=1.720000, pitch=54, velocity=25), 
    Note(start=0.952000, end=1.744000, pitch=42, velocity=25), 
    Note(start=0.952000, end=1.720000, pitch=47, velocity=29), 
    Note(start=1.384000, end=1.944000, pitch=62, velocity=41), 
    Note(start=1.384000, end=1.968000, pitch=59, velocity=29), 
    Note(start=1.368000, end=1.952000, pitch=35, velocity=33), 
    Note(start=1.688000, end=2.184000, pitch=50, velocity=37), 
    Note(start=1.720000, end=2.184000, pitch=54, velocity=37), 
    Note(start=1.744000, end=2.208000, pitch=42, velocity=29), 
    Note(start=1.720000, end=2.216000, pitch=47, velocity=21), 
    Note(start=1.944000, end=2.384000, pitch=62, velocity=41), 
    Note(start=1.968000, end=2.376000, pitch=59, velocity=9), 
    Note(start=1.952000, end=2.392000, pitch=35, velocity=29), 
    Note(start=2.184000, end=2.664000, pitch=50, velocity=29), 
    Note(start=2.216000, end=2.664000, pitch=47, velocity=17), 
    Note(start=2.208000, end=2.664000, pitch=42, velocity=33), 
    Note(start=2.184000, end=2.656000, pitch=54, velocity=37), 
    Note(start=2.384000, end=2.872000, pitch=62, velocity=33), 
    Note(start=2.376000, end=3.344000, pitch=59, velocity=29), 
    Note(start=2.392000, end=2.856000, pitch=35, velocity=33),
    #need to experiment with longer pauses between notes
    Note(start=4.8, end=5.8, pitch=40, velocity=37)]
]


sample_note_sequence1 = [
    [Note(start=0.928000, end=1.720000, pitch=54, velocity=25), 
    Note(start=0.952000, end=1.744000, pitch=42, velocity=25), 
    Note(start=0.952000, end=1.720000, pitch=47, velocity=29), 
    Note(start=1.384000, end=1.944000, pitch=62, velocity=41), 
    Note(start=1.384000, end=1.968000, pitch=59, velocity=29)]
        ]


sample_note_sequence2 = [Note(start=535.864000, end=536.296000, pitch=81, velocity=77), 
    Note(start=535.968000, end=536.368000, pitch=88, velocity=89), 
    Note(start=536.080000, end=536.664000, pitch=79, velocity=77), 
    Note(start=536.176000, end=537.056000, pitch=78, velocity=89), 
    Note(start=536.296000, end=537.608000, pitch=81, velocity=73), 
    Note(start=536.368000, end=543.224000, pitch=88, velocity=77), 
    Note(start=536.472000, end=536.752000, pitch=86, velocity=77), 
    Note(start=536.568000, end=537.416000, pitch=76, velocity=77), 
    Note(start=536.664000, end=537.984000, pitch=79, velocity=73), 
    Note(start=536.752000, end=544.648000, pitch=86, velocity=69)]
    

def main():
    sample = [sample_note_sequence2]
    v_sample = vectorize(sample_note_sequence2)

    encoder = SequenceEncoder(n_time_shift_events = 125, n_velocity_events = 32)
    assert encoder.n_events == 413
    encoded = encoder.encode_sequences([v_sample])
    decoded = encoder.decode_sequences(encoded)
    
    original_seq = sorted(sample[0], key = lambda x: x.start)
    decoded_seq = sorted(decoded[0], key = lambda x: x.start)
    for o,d in zip(original_seq, decoded_seq):
        try:
            assert o.start == d.start
            assert o.end == d.end
            assert o.pitch == d.pitch
            assert o.velocity == d.velocity
        except AssertionError:
            print("Encoding/Decoding error detected!")
            print("Original note:")
            print(o)
            print("Decoded encoded note:")
            print(d)
            print('************')
    print("Successful encoding and decoding of sequence!")


if __name__ == "__main__":
    main()

