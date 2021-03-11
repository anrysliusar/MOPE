import math
from random import randint
import numpy as np

numOfExperiments = 5
probList = (0.99, 0.98, 0.95, 0.90)
rkrTable = {2: (1.73, 1.72, 1.71, 1.69),
            6: (2.16, 2.13, 2.10, 2.00),
            8: (2.43, 4.37, 2.27, 2.17),
            10: (2.62, 2.54, 2.41, 2.29),
            12: (2.75, 2.66, 2.52, 2.39),
            15: (2.9, 2.8, 2.64, 2.49),
            20: (3.08, 2.96, 2.78, 2.62)}


def main():
    minLimY, maxLimY = 0, 100
    X1Min, X1MinNorm = -30, -1
    X1Max, X1MaxNorm = 20, 1
    X2Min, X2MinNorm = -70, -1
    X2Max, X2MaxNorm = -10, 1

    def checkRegression():
        normY1 = round(b0 - b1 - b2, 1)
        normY2 = round(b0 + b1 - b2, 1)
        normY3 = round(b0 - b1 + b2, 1)
        if normY1 == avgYArr[0] and normY2 == avgYArr[1] and normY3 == avgYArr[2]:
            return True
        else:
            return False

    def checkRegressionStr():
        if checkRegression():
            return "Значення перевірки нормаваного рівняння регресії сходяться"
        else:
            return "Значення перевірки нормаваного рівняння регресії не сходяться"

    def naturalRegress(x1, x2):
        return aa0 + aa1 * x1 + aa2 * x2

    def homogeneousDispersion():
        m = min(rkrTable, key=lambda x: abs(x - numOfExperiments))
        p = 0
        for ruv in (Ruv1, Ruv2, Ruv3):
            if ruv > rkrTable[m][0]:
                raise Exception("Потрібно більше експерементів")
            for rkr in range(len(rkrTable[m])):
                if ruv < rkrTable[m][rkr]:
                    p = rkr
        return probList[p]

    def calcFuv(u, v):
        if u >= v:
            return u / v
        else:
            return v / u

    print(checkRegressionStr())

    yMatrix = [[randint(minLimY, maxLimY) for i in range(numOfExperiments)] for j in range(3)]

    avgYArr = [sum(yMatrix[i][j] for j in range(numOfExperiments)) / numOfExperiments for i in range(3)]

    for i in range(3):
        print(f"Y{i + 1}: {yMatrix[i]}, Середне значення: {avgYArr[i]}")

    sigma1 = sum([(j - avgYArr[0]) ** 2 for j in yMatrix[0]]) / numOfExperiments
    sigma2 = sum([(j - avgYArr[1]) ** 2 for j in yMatrix[1]]) / numOfExperiments
    sigma3 = sum([(j - avgYArr[2]) ** 2 for j in yMatrix[2]]) / numOfExperiments
    sigmaTeta = math.sqrt((2 * (2 * numOfExperiments - 2)) / (numOfExperiments * (numOfExperiments - 4)))

    print(f"\nσ² y1: {round(sigma1, 3)}\nσ² y2: {round(sigma2, 3)}\nσ² y3: {round(sigma3, 3)}")
    print(f'Основне відхилення: {round(sigmaTeta, 5)}')

    Fuv1 = calcFuv(sigma1, sigma2)
    Fuv2 = calcFuv(sigma3, sigma1)
    Fuv3 = calcFuv(sigma3, sigma2)

    print(f"\nFuv1 = {round(Fuv1, 5)}\nFuv2 = {round(Fuv2, 5)}\nFuv3 = {round(Fuv3, 5)}\n")

    tetaUV1 = ((numOfExperiments - 2) / numOfExperiments) * Fuv1
    tetaUV2 = ((numOfExperiments - 2) / numOfExperiments) * Fuv2
    tetaUV3 = ((numOfExperiments - 2) / numOfExperiments) * Fuv3

    print(f"θuv1 = {round(tetaUV1, 5)}\nθuv2 = {round(tetaUV2, 5)}\nθuv3 = {round(tetaUV3, 5)}\n")

    Ruv1 = abs(tetaUV1 - 1) / sigmaTeta
    Ruv2 = abs(tetaUV2 - 1) / sigmaTeta
    Ruv3 = abs(tetaUV3 - 1) / sigmaTeta

    print(f"Ruv1 = {round(Ruv1, 5)}\nRuv2 = {round(Ruv2, 5)}\nRuv3 = {round(Ruv3, 5)}\n")
    print(f"Однорідна дисперсія: {homogeneousDispersion()}")

    mX1 = (X1MinNorm + X1MaxNorm + X1MinNorm) / 3
    mX2 = (X2MinNorm + X2MinNorm + X2MaxNorm) / 3
    mY = sum(avgYArr) / 3

    print(f"\nmx1: {round(mX1, 5)}\nmx2: {round(mX2, 5)}\nmy: {round(mY, 5)}")

    a1 = (X1MinNorm ** 2 + X1MaxNorm ** 2 + X1MinNorm ** 2) / 3
    a2 = (X1MinNorm * X2MinNorm + X1MaxNorm * X2MinNorm + X1MinNorm * X2MaxNorm) / 3
    a3 = (X2MinNorm ** 2 + X2MinNorm ** 2 + X2MaxNorm ** 2) / 3
    a11 = (X1MinNorm * avgYArr[0] + X1MaxNorm * avgYArr[1] + X1MinNorm * avgYArr[2]) / 3
    a22 = (X2MinNorm * avgYArr[0] + X2MinNorm * avgYArr[1] + X2MaxNorm * avgYArr[2]) / 3

    print(f"\na1: {round(a1, 5)}\na2: {round(a2, 5)}\na3: {round(a3, 5)}\na11: {round(a11, 5)}\na22: {round(a22, 5)}")

    b0 = np.linalg.det(np.dot([[mY, mX1, mX2],
                               [a11, a1, a2],
                               [a22, a2, a3]],
                              np.linalg.inv([[1, mX1, mX2],
                                             [mX1, a1, a2],
                                             [mX2, a2, a3]])))

    b1 = np.linalg.det(np.dot([[1, mY, mX2],
                               [mX1, a11, a2],
                               [mX2, a22, a3]],
                              np.linalg.inv([[1, mX1, mX2],
                                             [mX1, a1, a2],
                                             [mX2, a2, a3]])))

    b2 = np.linalg.det(np.dot([[1, mX1, mY],
                               [mX1, a1, a11],
                               [mX2, a2, a22]],
                              np.linalg.inv([[1, mX1, mX2],
                                             [mX1, a1, a2],
                                             [mX2, a2, a3]])))

    print(f"\nb0: {round(b0, 5)}\nb1: {round(b1, 5)}\nb2: {round(b2, 5)}")

    dX1 = math.fabs(X1Max - X1Min) / 2
    dX2 = math.fabs(X2Max - X2Min) / 2
    x10 = (X1Max + X1Min) / 2
    x20 = (X2Max + X2Min) / 2

    aa0 = b0 - b1 * x10 / dX1 - b2 * x20 / dX2
    aa1 = b1 / dX1
    aa2 = b2 / dX2

    print("\nНатуралізація коефіцієнтів:")
    print(f"dx1: {dX1}\ndx2: {dX2}")
    print(f"x10: {x10}\nx20: {x20}")
    print(f"\na0: {round(aa0, 5)}\na1: {round(aa1, 5)}\na2: {round(aa2, 5)}\n")

    try:
        homogeneousDispersion()
    except Exception as e:
        print(e)
        print("Збільшуємо кількість експерементів")
        numOfExperiments += 1
        return main()

    naturalRegress = [round(naturalRegress(X1Min, X2Min), 2),
                      round(naturalRegress(X1Max, X2Min), 2),
                      round(naturalRegress(X1Min, X2Max), 2)]
    print(f"Натуралізоване рівняння регресії: \n{naturalRegress}")

    if naturalRegress == avgYArr:
        print("Коефіцієнти натуралізованого рівняння регресії вірні")
    else:
        print("Коефіцієнти натуралізованого рівняння регресії не вірні")
    print(checkRegressionStr())

if __name__ == '__main__':
    main()
