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
model.load_weights("convThree99.h5")


score = 0
opp = 0
for count in range(100000):
    path, answer = findPathAnswer(pathR)

    letter, p = refineLetter(path, conv=True)
    if p == 1:
        continue

    trainX = np.array([letter])
    trainY = np.array([answer])

    result = model.predict(trainX)

    opp += 1
    t1 = np.where(result == result.max())
    t2 = np.where(trainY == trainY.max())
    if t1 == t2:
        score += 1

    print("\r", count, t1[1][0], score / opp, end="")











#-------------------------------------------------------------------
