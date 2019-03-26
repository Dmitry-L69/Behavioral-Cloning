import csv
import cv2
from tqdm import tqdm
import numpy as np
import random
import os


def generate_data(path, batch_size, validation, valid_prop):
    batch_size = batch_size // 6
    csv_lines = []
    with open(path) as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            csv_lines.append(line)

    random.shuffle(csv_lines)

    mid_point = len(csv_lines) - np.int32(len(csv_lines) * valid_prop)
    train = csv_lines[0:mid_point]

    valid = csv_lines[mid_point:len(csv_lines)]

    if validation:
        lines = valid
    else:
        lines = train

    offset = 0
    while True:
        images = []
        values = []

        if (offset + batch_size) > len(lines):
            offset = 0
            random.shuffle(lines)

        for i in range(offset, offset + batch_size):
            line = lines[i]
            center_path = line[0]
            left_path = line[1]
            right_path = line[2]
            center_value = np.float64(line[3])

            
            left_value = center_value + 0.2
            right_value = center_value - 0.2

            center_image = cv2.imread(center_path)
            center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
            left_image = cv2.imread(left_path)
            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
            right_image = cv2.imread(right_path)
            right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
            fliped_center_image = cv2.flip(center_image.copy(), 0)
            fliped_left_image = cv2.flip(left_image.copy(), 0)
            fliped_right_image = cv2.flip(right_image.copy(), 0)

            images.append(center_image)
            images.append(left_image)
            images.append(right_image)
            images.append(fliped_center_image)
            images.append(fliped_left_image)
            images.append(fliped_right_image)

            values.append(center_value)
            values.append(left_value)
            values.append(right_value)
            values.append(-1 * center_value)
            values.append(-1 * left_value)
            values.append(-1 * right_value)
        x = np.array(images)
        y = np.array(values)
        offset += batch_size
        yield x, y


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Conv2D, MaxPooling2D, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: ((x / 255.0) - 0.5), input_shape=(160,320,3)))
# model.add(Flatten(input_shape=(160,320,3)))
# model.add(Dropout(rate=0.7))
# model.add(Dense(400))
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.7))
# model.add(Dense(64))
# model.add(Dense(1))

model.add(Cropping2D(((70,25), (0,0))))
model.add(Conv2D(24, (5,5), subsample=(2,2), activation="relu"))
model.add(Conv2D(36, (5,5), subsample=(2,2), activation="relu"))
model.add(Conv2D(48, (5,5), subsample=(2,2), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='Adam')

batch_size = 128

model.fit_generator(generate_data('./video/driving_log.csv', batch_size, False, 0.2),validation_data=generate_data('./video/driving_log.csv', batch_size, True, 0.2),validation_steps=(len(os.listdir('./video/IMG')) * 2 * 0.2) // batch_size, verbose=1, steps_per_epoch=(len(os.listdir('./video/IMG')) * 2 * 0.8) // batch_size, epochs=10)
model.save('model.h5')

