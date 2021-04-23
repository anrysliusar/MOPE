import random
import sklearn.linear_model as lm
from scipy.stats import f, t
from functools import partial
from pyDOE2 import *
from time import time


def regression(x, b):
    y = sum([x[i] * b[i] for i in range(len(x))])
    return y


def skv(y, avgY, n, m):
    res = []
    for i in range(n):
        s = sum([(avgY[i] - y[i][j]) ** 2 for j in range(m)]) / m
        res.append(round(s, 3))
    return res


def planMatrix5(n, m):
    print(f'\nГенеруємо матрицю планування для n = {n}, m = {m}')

    y = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            y[i][j] = random.randint(minY, maxY)

    if n > 14:
        no = n - 14
    else:
        no = 1
    normX = ccdesign(3, center=(0, no))
    normX = np.insert(normX, 0, 1, axis=1)

    for i in range(4, 11):
        normX = np.insert(normX, i, 0, axis=1)

    l = 1.215

    for i in range(len(normX)):
        for j in range(len(normX[i])):
            if normX[i][j] < -1 or normX[i][j] > 1:
                if normX[i][j] < 0:
                    normX[i][j] = -l
                else:
                    normX[i][j] = l

    def addSqNums(x):
        for i in range(len(x)):
            x[i][4] = x[i][1] * x[i][2]
            x[i][5] = x[i][1] * x[i][3]
            x[i][6] = x[i][2] * x[i][3]
            x[i][7] = x[i][1] * x[i][3] * x[i][2]
            x[i][8] = x[i][1] ** 2
            x[i][9] = x[i][2] ** 2
            x[i][10] = x[i][3] ** 2
        return x

    normX = addSqNums(normX)

    x = np.ones(shape=(len(normX), len(normX[0])), dtype=np.int64)
    for i in range(8):
        for j in range(1, 4):
            if normX[i][j] == -1:
                x[i][j] = rangeX[j - 1][0]
            else:
                x[i][j] = rangeX[j - 1][1]

    for i in range(8, len(x)):
        for j in range(1, 3):
            x[i][j] = (rangeX[j - 1][0] + rangeX[j - 1][1]) / 2

    dx = [rangeX[i][1] - (rangeX[i][0] + rangeX[i][1]) / 2 for i in range(3)]

    x[8][1] = l * dx[0] + x[9][1]
    x[9][1] = -l * dx[0] + x[9][1]
    x[10][2] = l * dx[1] + x[9][2]
    x[11][2] = -l * dx[1] + x[9][2]
    x[12][3] = l * dx[2] + x[9][3]
    x[13][3] = -l * dx[2] + x[9][3]

    x = addSqNums(x)

    print('\nX:\n', x)
    print('\nX нормоване:\n')
    for i in normX:
        print([round(x, 2) for x in i])
    print('\nY:\n', y)

    return x, y, normX


def findCoef(X, Y, norm=False):
    skm = lm.LinearRegression(fit_intercept=False)
    skm.fit(X, Y)
    B = skm.coef_

    if norm == 1:
        print('\nКоефіцієнти рівняння регресії з нормованими X:')
    else:
        print('\nКоефіцієнти рівняння регресії:')
    B = [round(i, 3) for i in B]
    print(B)
    print('\nРезультат рівняння зі знайденими коефіцієнтами:\n', np.dot(X, B))
    return B


def kriteriaCochrana(y, avgY, n, m):
    f1 = m - 1
    f2 = n
    q = 0.05
    Skv = skv(y, avgY, n, m)
    Gp = max(Skv) / sum(Skv)
    print('\nПеревірка за критерієм Кохрена')
    return Gp


def cohren(f1, f2, q=0.05):
    q1 = q / f1
    fisherValue = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
    return fisherValue / (fisherValue + f1 - 1)


def bs(x, avgY, n):  # метод для оцінки коефіцієнтів
    res = [sum(1 * y for y in avgY) / n]

    for i in range(len(x[0])):
        b = sum(j[0] * j[1] for j in zip(x[:, i], avgY)) / n
        res.append(b)
    return res


def studentKriteria(x, y, avgY, n, m):
    Skv = skv(y, avgY, n, m)
    avgSkv = sum(Skv) / n

    s_Bs = (avgSkv / n / m) ** 0.5
    Bs = bs(x, avgY, n)
    ts = [round(abs(B) / s_Bs, 3) for B in Bs]

    return ts


def kriteriaFisher(y, avgY, newY, n, m, d):
    Sad = m / (n - d) * sum([(newY[i] - avgY[i]) ** 2 for i in range(len(y))])
    Skv = skv(y, avgY, n, m)
    avgSkv = sum(Skv) / n

    return Sad / avgSkv


def check(X, Y, B, n, m):
    print('\n\tПеревірка рівняння:')
    f1 = m - 1
    f2 = n
    f3 = f1 * f2
    q = 0.05

    student = partial(t.ppf, q=1 - q)
    Tstudent = student(df=f3)

    Gkr = cohren(f1, f2)

    avgY = [round(sum(i) / len(i), 3) for i in Y]
    print('\nСереднє значення y:', avgY)

    disp = skv(Y, avgY, n, m)
    print('Дисперсія y:', disp)

    Gp = kriteriaCochrana(Y, avgY, n, m)
    print(f'Gp = {Gp}')
    if Gp < Gkr:
        print(f'З ймовірністю {1 - q} дисперсії однорідні.')
    else:
        print("Необхідно збільшити кількість дослідів")
        m += 1
        main(n, m)

    ts = studentKriteria(X[:, 1:], Y, avgY, n, m)
    print('\nКритерій Стьюдента:\n', ts)
    res = [t for t in ts if t > Tstudent]
    finalK = [B[i] for i in range(len(ts)) if ts[i] in res]
    print('\nКоефіцієнти {} статистично незначущі, тому ми виключаємо їх з рівняння.'.format(
        [round(i, 3) for i in B if i not in finalK]))

    newY = []
    for j in range(n):
        newY.append(regression([X[j][i] for i in range(len(ts)) if ts[i] in res], finalK))

    print(f'\nЗначення "y" з коефіцієнтами {finalK}')
    print(newY)

    d = len(res)
    if d >= n:
        print('\nF4 <= 0')
        print('')
        return
    f4 = n - d

    Fp = kriteriaFisher(Y, avgY, newY, n, m, d)

    fisher = partial(f.ppf, q=0.95)
    Ft = fisher(dfn=f4, dfd=f3)
    print('\nПеревірка адекватності за критерієм Фішера')
    print('Fp =', Fp)
    print('Ft =', Ft)
    if Fp < Ft:
        print('Математична модель адекватна експериментальним даним')
    else:
        print('Математична модель не адекватна експериментальним даним')


def main(n, m):
    X5, Y5, normX5 = planMatrix5(n, m)

    averY5 = [round(sum(i) / len(i), 3) for i in Y5]
    B5 = findCoef(X5, averY5)

    check(normX5, Y5, B5, n, m)


if __name__ == '__main__':
    # Варіант - 120
    rangeX = ((-10, 3), (-7, 2), (-1, 6))

    avgMaxX = sum([x[1] for x in rangeX]) / 3
    avgMinX = sum([x[0] for x in rangeX]) / 3

    maxY = 200 + int(avgMaxX)
    minY = 200 + int(avgMinX)

    n = 15
    m = 3
    main(n, m)
