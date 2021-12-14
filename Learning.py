import os
import random
import tensorflow as tf
import numpy as np

def printList(list):
    for i in list:
        for j in i:
            print(j, end="")
        print()

def listY(list):
    for i, cnt in zip(list, range(len(list))):
        if i != [0]*len(list):
            yu = cnt
            break

    for i, cnt in zip(reversed(list), reversed(range(len(list)))):
        if i != [0]*len(list):
            yd = cnt
            break

    y = (yu + yd) / 2

    dy = int(y - len(list) / 2)
    return dy

def listYm(list, y):
    for i, cnt in zip(reversed(list[:-1*y]), reversed(range(y, len(list)))):
        list[cnt] = i
    for i in range(y):
        list[i] = [0] * len(list[0])
    return list

def listYp(list, y):
    for i, cnt in zip(list[y:], range(len(list) - y)):
        list[cnt] = i
    for i in range(len(list)-y, len(list)):
        list[i] = [0] * len(list[0])
    return list

def listX(list):
    ch = 0
    for i in range(len(list)):
        for j in range(len(list)):
            if list[j][i] != 0:
                ch = 1
                break
        if ch == 1:
            x1 = i
            break

    ch = 0
    for i in reversed(range(len(list))):
        for j in range(len(list)):
            if list[j][i] != 0:
                ch = 1
                break
        if ch == 1:
            x2 = i
            break

    x = (x1 + x2) / 2

    dx = int(x - len(list) / 2)
    return dx

def listXm(list, x):
    for i, cnt in zip(reversed(range(len(list)-x)), reversed(range(x, len(list)))):
        for j in range(len(list)):
            list[j][cnt] = list[j][i]

    for i in range(x):
        for j in range(len(list)):
            list[j][i] = 0
    return list

def listXp(list, x):
    for i, cnt in zip(range(x, len(list)), range(len(list) - x)):
        for j in range(len(list)):
            list[j][cnt] = list[j][i]
    for i in range(len(list)-x, len(list)):
        for j in range(len(list)):
            list[j][i] = 0
    return list

def correctY(list):
    dy = listY(list)
    if dy < 0:
        list = listYm(list, -1 * dy)
    else:
        list = listYp(list, dy)
    return list

def correctX(list):
    dx = listX(list)
    if dx < 0:
        list = listXm(list, -1 * dx)
    else:
        list = listXp(list, dx)
    return list

def findPathAnswer(path):
    directory = os.listdir(path)
    answerT = random.randrange(len(directory))
    path += directory[answerT] + "/"

    answer = np.zeros(len(directory))
    answer[answerT] = 1.

    directory = os.listdir(path)
    path += random.choice(directory)

    return [path, answer]

def refineLetter(path, conv=False):
    p = 0

    f = open(path, "r")
    letter = f.read()

    letter = letter.split("\n\n")
    letter = random.choice(letter)
    letter = letter.split("\n")

    for i, cnt in zip(letter, range(len(letter))):
        temp = []
        for j in i:
            temp.append(int(j))
        letter[cnt] = temp

    try:
        letter = correctY(letter)
        letter = correctX(letter)
    except:
        p = 1

    if conv == True:
        for i in range(len(letter)):
            for j in range(len(letter[0])):
                letter[i][j] = [letter[i][j]]

    letter = np.array(letter)

    if conv == False:
        letter = np.reshape(letter, -1)

    return letter, p

def setData(pathR, Count=10000, conv=False):
    p = 0
    trainX = []
    trainY = []
    for i in range(Count):
        print("\r", i, "/", Count, end="")
        path, answer = findPathAnswer(pathR)

        letter, pTemp = refineLetter(path, conv=conv)

        p = p or pTemp
        if p == 1:
            p = 0
            continue

        trainX.append(letter)
        trainY.append(answer)

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return [trainX, trainY]

pathR = "C:/Users/good2/Downloads/프로그램 모음/글자인식 인공지능/three/"
N = len(os.listdir(pathR))
w = 60
h = 60
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(60, (3, 3), input_shape=(w, h, 1), activation='relu'),
    tf.keras.layers.Conv2D(60, (3, 3), input_shape=(w, h, 60), activation='relu'),
    tf.keras.layers.Conv2D(60, (3, 3), input_shape=(w, h, 60), activation='relu'),
    tf.keras.layers.Conv2D(60, (3, 3), input_shape=(w, h, 60), activation='relu'),
    tf.keras.layers.Conv2D(60, (3, 3), input_shape=(w, h, 60), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(608, activation='relu'),
    tf.keras.layers.Dense(304, activation='relu'),
    tf.keras.layers.Dense(152, activation='relu'),
    tf.keras.layers.Dense(76, activation='relu'),
    tf.keras.layers.Dense(38, activation='relu'),
    tf.keras.layers.Dense(N, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights("convThree95.9.h5")

for i in range(10000):
    trainX, trainY = setData(pathR, Count=50000, conv=True)
    print("데이터 셋 준비가 완료되었습니다.")

    print("학습이 시작됩니다.")
    model.fit(trainX, trainY, epochs=13, batch_size=400)
    print("학습이 완료되었습니다.")

    model.save_weights('convThree' + str(i) + '.h5', save_format='h5')

