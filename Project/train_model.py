from music21 import converter, instrument, note, chord
import glob
import pickle
import keras 
import numpy
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint

def prepare_sequences(notes, n_vocab):
    sequence_length = 100   #det här kan vi gärna ändra
    pitchnames = sorted(set(item for item in notes))
    
    note_to_int = dict((note,number) for number,note in enumerate(pitchnames))
    
    network_input = []
    network_output = []
    
    for i in range(0,len(notes)-sequence_length,1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i+sequence_length]
        
        network_input.append([note_to_int[char] for char in sequence_in ])
        
        network_output.append(note_to_int[sequence_out])
        
    n_patterns = len(network_input)
    
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length,1))
    
    network_input = network_input /float(n_vocab)
    
    network_output = np_utils.to_categorical(network_output)
    
    
    return(network_input, network_output)

def get_notes():
    notes=[]
    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None
        
        parts = instrument.partitionByInstrument(midi)



        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes
            
            
        for element in notes_to_parse:
            #den ena är accord, t.ex. B5, den andra är siffror, t.ex 11.2
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                #print('.'.join(str(n) for n in element.normalOrder))
                

        with open('data/notes','wb') as filepath:
            pickle.dump(notes, filepath)
            
    return notes

def create_network(network_input, n_vocab):
    
    model = Sequential()
    model.add(LSTM(512,input_shape=(network_input.shape[1],network_input.shape[2]),return_sequences=True))    
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
    
    return model

def train(model,network_input,network_output):
    filepath = "weights-improvement-{epoch:02d}--{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor = 'loss',
        verbose = 0,
        save_best_only=True,
        mode='min'
    )
    
    callbacks_list = [checkpoint]
    
    model.fit(network_input, network_output, epochs=10,batch_size = 64, callbacks=callbacks_list)
    
    
    return

notes = get_notes()
n_vocab = len(set(notes))
network_input,network_output = prepare_sequences(notes,n_vocab)

model = create_network(network_input,n_vocab)
train(model,network_input,network_output)
