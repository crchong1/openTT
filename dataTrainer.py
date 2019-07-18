import pandas as pd
import os
import librosa
import librosa.display
from wavfilehelper import WavFileHelper
wavfilehelper = WavFileHelper()
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 

# audiodata = []
# for index, row in metadata.iterrows():
    
#     file_name = os.path.join(os.path.abspath('/UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
#     data = wavfilehelper.read_file_properties(file_name)
#     audiodata.append(data)

# # Convert into a Panda dataframe
# audiodf = pd.DataFrame(audiodata, columns=['num_channels','sample_rate','bit_depth'])

def main():
    # Set the path to the full UrbanSound dataset 
    fulldatasetpath = '/Urban Sound/UrbanSound8K/audio/'
    metadata = pd.read_csv(fulldatasetpath + '../metadata/UrbanSound8K.csv')
    features = []

    # Iterate through each sound file and extract the features 
    for index, row in metadata.iterrows():
        file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
        class_label = row["class_name"]
        data = extract_features(file_name)
        features.append([data, class_label])

    # Convert into a Panda dataframe 
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
    print('Finished feature extraction from ', len(featuresdf), ' files')


    # Convert features and corresponding classification labels into numpy arrays
    trainX = np.array(featuresdf.feature.tolist())
    labelsY = np.array(featuresdf.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    categoricalLabels = to_categorical(le.fit_transform(labelsY)) 

    # split the dataset 

    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
    kf = KFold(n_splits=10, shuffle=False)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None 
    return mfccsscaled
    
    




num_rows = 40
num_columns = 174
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2

def model():
    # Construct model 
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_labels, activation='softmax'))
    return model

def compile(model):
    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # Display model architecture summary 
    model.summary()

    # Calculate pre-training accuracy 
    score = model.evaluate(x_test, y_test, verbose=1)
    accuracy = 100*score[1]

    print("Pre-training accuracy: %.4f%%" % accuracy) 
    return model


num_epochs = 72
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])

if __name__ == "__main__":
    main()