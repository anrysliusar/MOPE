import numpy as np
import random
from numpy.linalg import solve
from scipy.stats import f, t
from functools import partial

rangeX = [(-30, 20), (-70, -10), (-70, -40)]
avgMaxX = (20 - 10 - 40) / 3
avgMinX = (-30 - 70 - 40) / 3

maxY = 200 + int(avgMaxX)
minY = 200 + int(avgMinX)


def matrixPlan(n, m):
    y = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            y[i][j] = random.randint(minY, maxY)
    NormalX = np.array([[1, -1, -1, -1],
                        [1, -1, 1, 1],
                        [1, 1, -1, 1],
                        [1, 1, 1, -1],
                        [1, -1, -1, 1],
                        [1, -1, 1, -1],
                        [1, 1, -1, -1],
                        [1, 1, 1, 1]])
    xNorm = NormalX[:len(y)]

    x = np.ones(shape=(len(xNorm), len(xNorm[0])))
    for i in range(len(xNorm)):
        for j in range(1, len(xNorm[i])):
            if xNorm[i][j] == -1:
                x[i][j] = rangeX[j - 1][0]
            else:
                x[i][j] = rangeX[j - 1][1]

    print('\nМатриця планування')
    print(np.concatenate((x, y), axis=1))

    return x, y, xNorm


def regress(x, b):
    y = sum([x[i] * b[i] for i in range(len(x))])
    return y


def coefOfRegress(x, avgY, n):
    mx1 = sum(x[:, 1]) / n
    mx2 = sum(x[:, 2]) / n
    mx3 = sum(x[:, 3]) / n
    my = sum(avgY) / n

    a1 = sum([avgY[i] * x[i][1] for i in range(len(x))]) / n
    a2 = sum([avgY[i] * x[i][2] for i in range(len(x))]) / n
    a3 = sum([avgY[i] * x[i][3] for i in range(len(x))]) / n

    a11 = sum([i ** 2 for i in x[:, 1]]) / n
    a12 = sum([x[i][1] * x[i][2] for i in range(len(x))]) / n
    a13 = sum([x[i][1] * x[i][3] for i in range(len(x))]) / n
    a23 = sum([x[i][2] * x[i][3] for i in range(len(x))]) / n
    a22 = sum([i ** 2 for i in x[:, 2]]) / n
    a33 = sum([i ** 2 for i in x[:, 3]]) / n

    X = [[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a23], [mx3, a13, a23, a33]]
    Y = [my, a1, a2, a3]
    B = [round(i, 2) for i in solve(X, Y)]

    print('\nРівняння регресії')
    print(f'{B[0]} + {B[1]} * x1 + {B[2]} * x2 + {B[3]} * x3')

    return B


def dispersion(y, avgY, n, m):
    res = []
    for i in range(n):
        s = sum([(avgY[i] - y[i][j]) ** 2 for j in range(m)]) / m
        res.append(s)
    return res


def testKohren(y, avgY, n, m):
    S_kv = dispersion(y, avgY, n, m)
    Gp = max(S_kv) / sum(S_kv)
    print('\nПеревірка за критерієм Кохрена')
    return Gp


def kohren(f1, f2, q=0.05):
    q1 = q / f1
    fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
    return fisher_value / (fisher_value + f1 - 1)


# оцінки коефіцієнтів
def bs(x, y, avgY, n):
    res = [sum(1 * y for y in avgY) / n]
    for i in range(3):  # 4 - ксть факторів
        b = sum(j[0] * j[1] for j in zip(x[:, i], avgY)) / n
        res.append(b)
    return res


def testStudent(x, y, avgY, n, m):
    Skv = dispersion(y, avgY, n, m)
    avg_skv = sum(Skv) / n

    s_Bs = (avg_skv / n / m) ** 0.5
    Bs = bs(x, y, avgY, n)
    ts = [abs(B) / s_Bs for B in Bs]

    return ts


def testFisher(y, avgY, y_new, n, m, d):
    S_ad = m / (n - d) * sum([(y_new[i] - avgY[i]) ** 2 for i in range(len(y))])
    S_kv = dispersion(y, avgY, n, m)
    avg_S_kv = sum(S_kv) / n

    return S_ad / avg_S_kv


def main(n, m):
    f1 = m - 1
    f2 = n
    f3 = f1 * f2
    q = 0.05

    student = partial(t.ppf, q=1 - 0.025)
    t_student = student(df=f3)

    G_kr = kohren(f1, f2)

    x, y, x_norm = matrixPlan(n, m)
    avgY = [round(sum(i) / len(i), 2) for i in y]

    B = coefOfRegress(x, avgY, n)

    Gp = testKohren(y, avgY, n, m)
    print(f'Gp = {Gp}')
    if Gp < G_kr:
        print(f'З ймовірністю {1 - q} дисперсії однорідні.')
    else:
        print("Необхідно збільшити кількість дослідів")
        m += 1
        main(n, m)

    ts = testStudent(x_norm[:, 1:], y, avgY, n, m)
    print('\nКритерій Стюдента:\n', ts)
    res = [t for t in ts if t > t_student]
    final_k = [B[ts.index(i)] for i in ts if i in res]
    print('Коефіцієнти {} статистично незначущі, тому ми виключаємо їх з рівняння.'.format(
        [i for i in B if i not in final_k]))

    yNew = []
    for j in range(n):
        yNew.append(regress([x[j][ts.index(i)] for i in ts if i in res], final_k))

    print(f'\nЗначення "y" з коефіцієнтами {final_k}')
    print(yNew)

    d = len(res)
    f4 = n - d
    Fp = testFisher(y, avgY, yNew, n, m, d)

    fisher = partial(f.ppf, q=1 - 0.05)
    ft = fisher(dfn=f4, dfd=f3)

    print('\nПеревірка адекватності за критерієм Фішера')
    print('Fp =', Fp)
    print('F_t =', ft)
    if Fp < ft:
        print('Математична модель адекватна експериментальним даним')
    else:
        print('Математична модель не адекватна експериментальним даним')


if __name__ == '__main__':
    n = 4  # кількість експериментів (рядків матриці планування)
    m = 4  # кількість вимірів y за однією й тією ж самою комбінації факторів
    main(n, m)
