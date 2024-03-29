from pydub import AudioSegment
from pydub.utils import make_chunks

myaudio = AudioSegment.from_file("pingpong.wav" , "wav") 
chunk_length_ms = 5000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
for i, chunk in enumerate(chunks):
    chunk_name = "chunk{0}.wav".format(i)
    chunk.export(chunk_name, format="wav")