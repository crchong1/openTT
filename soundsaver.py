import sounddevice as sd
import numpy as np 
from pymongo import MongoClient
import datetime
import ssl
samplerate = 22050
sd.default.samplerate = samplerate
sd.default.channels = 1
duration = 5
current_time = str(datetime.datetime.now)
myrecording = sd.rec(int(duration * samplerate))
sd.wait()
client = MongoClient("mongodb+srv://user:mongo@cluster0-d8nub.mongodb.net/test?retryWrites=true&w=majority", ssl=True, ssl_cert_reqs=ssl.CERT_NONE)
print(client.list_database_names())
db = client['opentabletennis']
collection = db['usage']
recording_arr = myrecording.tolist()
entry = {
    "time": current_time,
    "sample": recording_arr
}

# pipeline: rasbpi gets data, 
collection.usage.insert_one(entry)