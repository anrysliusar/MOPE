import numpy as np
import random
from scipy.stats import f, t
from numpy.linalg import solve
import sklearn.linear_model as lm


def main(n, m):
    main1 = linear(n, m)
    if not main1:
        interactionEffect = withInteractionEffect(n, m)
        if not interactionEffect:
            main(n, m)


def linear(n, m):
    f1 = m - 1
    f2 = n
    f3 = f1 * f2
    q = 0.05

    x, y, normX = linearPlanMatrix(n, m, rangeX)

    avgY, B = regressionEquation(x, y, n)

    dispersionArr = findDispersion(y, avgY, n, m)

    tempCohren = f.ppf(q=(1 - q / f1), dfn=f2, dfd=(f1 - 1) * f2)
    cohrenCrTable = tempCohren / (tempCohren + f1 - 1)
    Gp = max(dispersionArr) / sum(dispersionArr)

    print('\nПеревірка за критерієм Кохрена:\n')
    print(f'Розрахункове значення: Gp = {Gp}'
          f'\nТабличне значення: Gt = {cohrenCrTable}')
    if Gp < cohrenCrTable:
        print(f'З ймовірністю {1 - q} дисперсії однорідні.')
    else:
        print("Необхідно збільшити ксть дослідів")
        m += 1
        linear(n, m)

    qq = (1 + 0.95) / 2
    studentCrTable = t.ppf(df=f3, q=qq)
    studentT = criteriaStudent(normX[:, 1:], avgY, n, m, dispersionArr)

    print('\nТабличне значення критерій Стьюдента:\n', studentCrTable)
    print('Розрахункове значення критерій Стьюдента:\n', studentT)
    resStudentT = [temp for temp in studentT if temp > studentCrTable]
    finalCoefficients = [B[studentT.index(i)] for i in studentT if i in resStudentT]
    print('Коефіцієнти {} статистично незначущі.'.
          format([i for i in B if i not in finalCoefficients]))

    newY = []
    for j in range(n):
        newY.append(
            findRegres([x[j][studentT.index(i)] for i in studentT if i in resStudentT], finalCoefficients))

    print(f'\nОтримаємо значення рівння регресії для {m} дослідів: ')
    print(newY)

    d = len(resStudentT)
    f4 = n - d
    Fp = criteriaFisher(y, avgY, newY, n, m, d, dispersionArr)
    Ft = f.ppf(dfn=f4, dfd=f3, q=1 - 0.05)

    print('\nПеревірка адекватності за критерієм Фішера:\n')
    print('Розрахункове значення критерія Фішера: Fp =', Fp)
    print('Табличне значення критерія Фішера: Ft =', Ft)
    if Fp < Ft:
        print('Математична модель адекватна експериментальним даним')
        return True
    else:
        print('Математична модель не адекватна експериментальним даним')
        return False



def findDispersion(y, avgY, n, m):
    result = []
    for i in range(n):
        s = sum([(avgY[i] - y[i][j]) ** 2 for j in range(m)]) / m
        result.append(round(s, 3))
    return result


def findRegres(x, b):
    y = sum([x[i] * b[i] for i in range(len(x))])
    return y


def interactionPlanMatrix(n, m):
    normalizedX = [[1, -1, -1, -1],
                    [1, -1, 1, 1],
                    [1, 1, -1, 1],
                    [1, 1, 1, -1],
                    [1, -1, -1, 1],
                    [1, -1, 1, -1],
                    [1, 1, -1, -1],
                    [1, 1, 1, 1]]
    y = np.zeros(shape=(n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            y[i][j] = random.randint(minY, maxY)
    for x in normalizedX:
        x.append(x[1] * x[2])
        x.append(x[1] * x[3])
        x.append(x[2] * x[3])
        x.append(x[1] * x[2] * x[3])
    normalizedX = np.array(normalizedX[:len(y)])
    x = np.ones(shape=(len(normalizedX), len(normalizedX)), dtype=np.int64)
    for i in range(len(normalizedX)):
        for j in range(1, 4):
            if normalizedX[i][j] == -1:
                x[i][j] = rangeX[j - 1][0]
            else:
                x[i][j] = rangeX[j - 1][1]
    for i in range(len(x)):
        x[i][4] = x[i][1] * x[i][2]
        x[i][5] = x[i][1] * x[i][3]
        x[i][6] = x[i][2] * x[i][3]
        x[i][7] = x[i][1] * x[i][3] * x[i][2]
    print(f'Матриця планування при n = {n} та m = {m}:')
    print('З кодованими значеннями:')
    print('\n     X0    X1    X2    X3  X1X2  X1X3  X2X3 X1X2X3   Y1    Y2     Y3')
    print(np.concatenate((x, y), axis=1))
    print('Нормовані значення:')
    print(normalizedX)
    return x, y, normalizedX


def findCoef(X, Y, isNorm=False):
    skm = lm.LinearRegression(fit_intercept=False)
    skm.fit(X, Y)
    coefficientsB = skm.coef_
    if isNorm == 1:
        print('Коефіцієнти з нормованими Х:')
    else:
        print('Коефіцієнти рівняння регресії')
    coefficientsB = [round(i, 3) for i in coefficientsB]
    print(coefficientsB)
    return coefficientsB


def bs(x, y, avgY, n):
    result = [sum(1 * y for y in avgY) / n]
    for i in range(7):
        b = sum(j[0] * j[1] for j in zip(x[:, i], avgY)) / n
        result.append(b)
    return result


def criteriaStudent2(x, y, avgY, n, m):
    studentSquared = findDispersion(y, avgY, n, m)
    avgStudentSquared = sum(studentSquared) / n
    studentsBs = (avgStudentSquared / n / m) ** 0.5
    Bs = bs(x, y, avgY, n)
    ts = [round(abs(B) / studentsBs, 3) for B in Bs]
    return ts


def criteriaStudent(x, avgY, n, m, dispersion):
    avgDispersion = sum(dispersion) / n
    studentsBeta = (avgDispersion / n / m) ** 0.5
    beta = [sum(1 * y for y in avgY) / n]
    for i in range(3):
        b = sum(j[0] * j[1] for j in zip(x[:, i], avgY)) / n
        beta.append(b)
    t = [round(abs(b) / studentsBeta, 3) for b in beta]
    return t


def criteriaFisher(y, avgY, newY, n, m, d, dispersion):
    Sad = m / (n - d) * sum([(newY[i] - avgY[i]) ** 2 for i in range(len(y))])
    avgDispersion = sum(dispersion) / n

    return Sad / avgDispersion


def check(X, Y, B, n, m):
    f1 = m - 1
    f2 = n
    f3 = f1 * f2
    q = 0.05
    avgY = [round(sum(i) / len(i), 3) for i in Y]
    print('Середнє знач. у: ', avgY)
    dispersionArr = findDispersion(Y, avgY, n, m)
    qq = (1 + 0.95) / 2

    studentsCriteriaTable = t.ppf(df=f3, q=qq)
    ts = criteriaStudent2(X[:, 1:], Y, avgY, n, m)

    tempCohren = f.ppf(q=(1 - q / f1), dfn=f2, dfd=(f1 - 1) * f2)
    cohrenCriteriaTable = tempCohren / (tempCohren + f1 - 1)
    Gp = max(dispersionArr) / sum(dispersionArr)
    print('Дисперсія: ', dispersionArr)
    print(f'Gp = {Gp}')
    if Gp < cohrenCriteriaTable:
        print(f'Дисперсії однорідні з ймовірністю {1 - q}')
    else:
        print(f'Дисперсія неоднорідна. Збільшуємо к-сть дослідів з {m} до {m + 1}')
        m += 1
        withInteractionEffect(n, m)

    print('\nКритерій Стьюдента:\n', ts)
    res = [t for t in ts if t > studentsCriteriaTable]
    finalK = [B[i] for i in range(len(ts)) if ts[i] in res]
    print('\nКоефіцієнти {} статистично незначущі, тому ми виключаємо їх з рівняння.'.format(
        [round(i, 3) for i in B if i not in finalK]))

    newY = []
    for j in range(n):
        newY.append(findDispersion([X[j][i] for i in range(len(ts)) if ts[i] in res], finalK))

    print(f'\nЗначення "y" з коефіцієнтами {finalK}')
    print(newY)

    d = len(res)
    if d >= n:
        print('\nF4 <= 0')
        print('')
        return
    f4 = n - d

    Fp = criteriaFisher(Y, avgY, newY, n, m, d, dispersionArr)

    Ft = f.ppf(dfn=f4, dfd=f3, q=1 - 0.05)

    print('\nПеревірка адекватності за критерієм Фішера')
    print('Fp =', Fp)
    print('Ft =', Ft)
    if Fp < Ft:
        print('Математична модель адекватна експериментальним даним')
        return True
    else:
        print('Математична модель не адекватна експериментальним даним')
        return False


def withInteractionEffect(n, m):
    X, Y, normalizedX = interactionPlanMatrix(n, m)

    avgY = [round(sum(i) / len(i), 3) for i in Y]

    normalizedB = findCoef(normalizedX, avgY, norm=True)

    return check(normalizedX, Y, normalizedB, n, m, norm=True)


def linearPlanMatrix(n, m, rangeX):
    normalizedX = np.array([[1, -1, -1, -1],
                             [1, -1, 1, 1],
                             [1, 1, -1, 1],
                             [1, 1, 1, -1],
                             [1, -1, -1, 1],
                             [1, -1, 1, -1],
                             [1, 1, -1, -1],
                             [1, 1, 1, 1]])
    y = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            y[i][j] = random.randint(minY, maxY)

    normalizedX = normalizedX[:len(y)]

    x = np.ones(shape=(len(normalizedX), len(normalizedX[0])))
    for i in range(len(normalizedX)):
        for j in range(1, len(normalizedX[i])):
            if normalizedX[i][j] == -1:
                x[i][j] = rangeX[j - 1][0]
            else:
                x[i][j] = rangeX[j - 1][1]

    print('\nМатриця планування:')
    print('\n    X0  X1   X2   X3   Y1   Y2   Y3  ')
    print(np.concatenate((x, y), axis=1))

    return x, y, normalizedX


def regressionEquation(x, y, n):
    avgY = [round(sum(i) / len(i), 2) for i in y]

    mx1 = sum(x[:, 1]) / n
    mx2 = sum(x[:, 2]) / n
    mx3 = sum(x[:, 3]) / n

    my = sum(avgY) / n

    a1 = sum([avgY[i] * x[i][1] for i in range(len(x))]) / n
    a2 = sum([avgY[i] * x[i][2] for i in range(len(x))]) / n
    a3 = sum([avgY[i] * x[i][3] for i in range(len(x))]) / n

    a12 = sum([x[i][1] * x[i][2] for i in range(len(x))]) / n
    a13 = sum([x[i][1] * x[i][3] for i in range(len(x))]) / n
    a23 = sum([x[i][2] * x[i][3] for i in range(len(x))]) / n

    a11 = sum([i ** 2 for i in x[:, 1]]) / n
    a22 = sum([i ** 2 for i in x[:, 2]]) / n
    a33 = sum([i ** 2 for i in x[:, 3]]) / n

    X = [[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a23], [mx3, a13, a23, a33]]
    Y = [my, a1, a2, a3]
    B = [round(i, 2) for i in solve(X, Y)]

    print('\nРівняння регресії:')
    print(f'y = {B[0]} + {B[1]}*x1 + {B[2]}*x2 + {B[3]}*x3')

    return avgY, B


if __name__ == '__main__':
    # Варіант - 120
    rangeX = ((15, 45), (-35, 15), (-35, -5))

    avgMinX = int(sum([x[0] for x in rangeX]) / 3)
    avgMaxX = int(sum([x[1] for x in rangeX]) / 3)

    minY = 200 + avgMinX
    maxY = 200 + avgMaxX

    n = 8
    m = 4
    main(n, m)
