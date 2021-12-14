import pygame as pg
import numpy as np
import tensorflow as tf
import copy

def phoneme(r1, r2, r3):
    letter = ord('가')
    letter += r1 * 588
    letter += r2 * 28
    letter += r3
    return chr(letter)

def printList(list):
    for i in list:
        for j in i:
            print(int(j), end="")
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

#인공지능 모델 받기
w = 60
h = 60
model1 = tf.keras.models.Sequential([
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
        tf.keras.layers.Dense(19, activation='softmax')
    ])
model2 = tf.keras.models.Sequential([
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
        tf.keras.layers.Dense(21, activation='softmax')
    ])
model3 = tf.keras.models.Sequential([
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
        tf.keras.layers.Dense(28, activation='softmax')
    ])
model1.load_weights("convOne99.4.h5")
model2.load_weights("convTwo99.6.h5")
model3.load_weights("convThree95.9.h5")

#파이 게임 초기화
pg.init()
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) #화면 크기 설정
clock = pg.time.Clock()

#변수
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CELL_SIZE = 10
COLUMN_COUNT = 60
ROW_COUNT = 60

ch = 0
count = 1
letter = np.full((60, 60), 0.)
letter = letter.tolist()
while count > 0:
    count += 1
    screen.fill(WHITE)

    #변수 업데이트
    event = pg.event.poll() #이벤트 처리
    if event.type == pg.QUIT:
        break
    elif event.type == pg.MOUSEBUTTONDOWN:
        ch = 1
    elif event.type == pg.MOUSEBUTTONUP:
        ch = 0
    if ch == 1 and (event.type == pg.MOUSEMOTION or event.type == pg.MOUSEBUTTONDOWN or event.type == pg.MOUSEBUTTONUP):
        y = event.pos[0] // CELL_SIZE
        x = event.pos[1] // CELL_SIZE
        try:
            size = 4
            for i in range(size):
                for j in range(size):
                    letter[x+int(i - size / 2)][y+int(j - size / 2)] = 1

        except IndexError:
            continue
    keys = pg.key.get_pressed()
    if (keys[pg.K_SPACE]):
        letter = np.full((60, 60), 0.)
        letter = letter.tolist()

    #화면 그리기
    for x in range(COLUMN_COUNT):
        for y in range(ROW_COUNT):
            if letter[y][x] == 1:
                pg.draw.rect(screen, BLACK, pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pg.display.flip()

    #인공지능 분석
    if (keys[pg.K_LSHIFT]):
        X = copy.deepcopy(letter)

        X = correctX(X)
        X = correctY(X)

        for i in range(w):
            for j in range(h):
                X[i][j] = [X[i][j]]
        X = np.array([X])

        result1 = model1.predict(X)
        result2 = model2.predict(X)
        result3 = model3.predict(X)

        result1 = np.where(result1 == result1.max())
        result2 = np.where(result2 == result2.max())
        result3 = np.where(result3 == result3.max())

        result1 = result1[1][0]
        result2 = result2[1][0]
        result3 = result3[1][0]

        print('\r', result1, result2, result3, phoneme(result1, result2, result3), end="")

pg.quit()