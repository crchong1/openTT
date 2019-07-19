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
import torchNN
from torchNN import CNN
import os
import librosa

# make cron job to do this every 25 seconds, which will make 1 sample every 30 sec, or 2 per minute
def recordAndUpload():
    client = MongoClient("mongodb+srv://user:mongo@cluster0-d8nub.mongodb.net/test?retryWrites=true&w=majority", ssl=True, ssl_cert_reqs=ssl.CERT_NONE)
    db = client['opentabletennis']
    myrecording = sd.rec(int(duration * samplerate))
    sd.wait()
    normalized_data = 2.*(myrecording - np.min(myrecording))/np.ptp(myrecording)-1 # normalize between -1 and 1
    usage_collection = db['usage']
    recording_arr = normalized_data.tolist()
    entry = {
        "time": current_time,
        "sample": recording_arr
    }
    usage_collection.insert_one(entry)

def isUsed():
    client = MongoClient("mongodb+srv://user:mongo@cluster0-d8nub.mongodb.net/test?retryWrites=true&w=majority", ssl=True, ssl_cert_reqs=ssl.CERT_NONE)
    db = client['opentabletennis']
    usage_collection = db['usage']
    all_documents = usage_collection.find({})
    isused_collection = db['isUsed']
    for document in all_documents:
        time = document["time"]
        sample = np.array(document["sample"])
        sample.shape = (1, 110250)
        result = torchNN.predict(sample)[0]
        answer = False
        if(result == 11):
            answer = True
        entry = {
            "time": time,
            "isUsed": answer
        }
        isused_collection.insert_one(entry)

def test():
    myrecording = sd.rec(int(duration * samplerate))
    sd.wait()
    allData = 2.*(myrecording - np.min(myrecording))/np.ptp(myrecording)-1 # normalize between -1 and 1
    allData.shape = (1, 110250)
    # allData = []
    # for filename in os.listdir("./samplePingPongSounds"):
    #     print(filename)
    #     data, rate = librosa.load("./samplePingPongSounds" + os.sep + filename, mono=True, duration=5.0)
    #     data = 2.*(data - np.min(data))/np.ptp(data)-1 # normalize between -1 and 1
    #     if(data.shape[0] < 110250):
    #         data = np.pad(data, (0,110250-data.shape[0]), mode="symmetric")
    #     allData.append(data)
    # allData = np.array(allData)
    print(allData.shape)
    print(torchNN.predict(allData))

if __name__ == "__main__":
    recordAndUpload()