import math
import numpy
import random
from _decimal import Decimal
from functools import reduce
from itertools import compress
from scipy.stats import f, t

# Ініціалізація змінних
xMin = [-30, -70, -70]
xMax = [20, -10, -40]
x0 = [(xMax[_] + xMin[_]) / 2 for _ in range(3)]
dx = [xMax[_] - x0[_] for _ in range(3)]
norm_plan_raw = [[-1, -1, -1],
                 [-1, +1, +1],
                 [+1, -1, +1],
                 [+1, +1, -1],
                 [-1, -1, +1],
                 [-1, +1, -1],
                 [+1, -1, -1],
                 [+1, +1, +1],
                 [-1.73, 0, 0],
                 [+1.73, 0, 0],
                 [0, -1.73, 0],
                 [0, +1.73, 0],
                 [0, 0, -1.73],
                 [0, 0, +1.73]]

naturalPlanRaw = [[xMin[0], xMin[1], xMin[2]],
                  [xMin[0], xMin[1], xMax[2]],
                  [xMin[0], xMax[1], xMin[2]],
                  [xMin[0], xMax[1], xMax[2]],
                  [xMax[0], xMin[1], xMin[2]],
                  [xMax[0], xMin[1], xMax[2]],
                  [xMax[0], xMax[1], xMin[2]],
                  [xMax[0], xMax[1], xMax[2]],
                  [-1.73 * dx[0] + x0[0], x0[1], x0[2]],
                  [1.73 * dx[0] + x0[0], x0[1], x0[2]],
                  [x0[0], -1.73 * dx[1] + x0[1], x0[2]],
                  [x0[0], 1.73 * dx[1] + x0[1], x0[2]],
                  [x0[0], x0[1], -1.73 * dx[2] + x0[2]],
                  [x0[0], x0[1], 1.73 * dx[2] + x0[2]],
                  [x0[0], x0[1], x0[2]]]


# Основні функції
def equationRegres(x1, x2, x3, coefficients, importance=[True] * 11):
    factors_array = [1, x1, x2, x3, x1 * x1, x2 * x2, x3 * x3, x1 * x2, x1 * x3, x2 * x3, x1 * x2 * x3]
    return sum([el[0] * el[1] for el in compress(zip(coefficients, factors_array), importance)])


def func(x1, x2, x3):
    coefficients = [2.1, 1.7, 6.8, 6.6, 9.5, 1.0, 3.9, 3.0, 0.1, 4.5, 1.8]
    return equationRegres(x1, x2, x3, coefficients)


def generateFactorsTable(rawArr):
    raw_list = [row + [row[0] * row[1], row[0] * row[2], row[1] * row[2], row[0] * row[1] * row[2]] + list(
        map(lambda x: x ** 2, row)) for row in rawArr]
    return list(map(lambda row: list(map(lambda el: round(el, 3), row)), raw_list))


def generateY(m, factorsTable):
    return [[round(func(row[0], row[1], row[2]) + random.randint(-5, 5), 3) for _ in range(m)] for row in factorsTable]


# Вивід результатів
def printMatrix(m, n, factors, valsY, additionalText=":"):
    labels_table = list(map(lambda x: x.ljust(10),
                            ["x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"] + [
                                "y{}".format(i + 1) for i in range(m)]))
    rows_table = [list(factors[i]) + list(valsY[i]) for i in range(n)]
    print("\nМатриця планування" + additionalText)
    print(" ".join(labels_table))
    print("\n".join([" ".join(map(lambda j: "{:<+10}".format(j), rows_table[i])) for i in range(len(rows_table))]))
    print("\t")


def printEquation(coefficients, importance=[True] * 11):
    XiNames = list(compress(["", "x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"], importance))
    coefficientsToPrint = list(compress(coefficients, importance))
    equation = " ".join(
        ["".join(i) for i in zip(list(map(lambda x: "{:+.2f}".format(x), coefficientsToPrint)), XiNames)])
    print("Рівняння регресії: y = " + equation)


def setFactorsTable(factorsTable):
    def x_i(i):
        withNullFactor = list(map(lambda x: [1] + x, generateFactorsTable(factorsTable)))
        res = [row[i] for row in withNullFactor]
        return numpy.array(res)

    return x_i


def Mij(*arrays):
    return numpy.average(reduce(lambda accum, el: accum * el, list(map(lambda el: numpy.array(el), arrays))))


def findCoeffs(factors, y_values):
    Xi = setFactorsTable(factors)
    coefficients = [[Mij(Xi(column), Xi(row)) for column in range(11)] for row in range(11)]
    numpyY = list(map(lambda row: numpy.average(row), y_values))
    freeVal = [Mij(numpyY, Xi(i)) for i in range(11)]
    betaCoeffs = numpy.linalg.solve(coefficients, freeVal)
    return list(betaCoeffs)


# Критерії
def cochranCriteria(m, n, y_table):
    def getCochranVal(f1, f2, q):
        part_result1 = q / f2
        params = [part_result1, f1, (f2 - 1) * f1]
        fisher = f.isf(*params)
        result = fisher / (fisher + (f2 - 1))
        return Decimal(result).quantize(Decimal('.0001')).__float__()

    print("Перевірка рівномірності дисперсій за критерієм Кохрена: m = {}, N = {}".format(m, n))
    variationsY = [numpy.var(i) for i in y_table]
    variationMaxY = max(variationsY)
    gp = variationMaxY / sum(variationsY)
    f1 = m - 1
    f2 = n
    p = 0.95
    q = 1 - p
    gt = getCochranVal(f1, f2, q)
    print("Gp = {}; Gt = {}; f1 = {}; f2 = {}; q = {:.2f}".format(gp, gt, f1, f2, q))
    if gp < gt:
        print("Gp < Gt => дисперсії рівномірні - все правильно")
        return True
    else:
        print("Gp > Gt => дисперсії нерівномірні - треба ще експериментів")
        return False


def studentCriteria(m, n, y_table, betaCoeffs):
    def getStudentVal(f3, q):
        return Decimal(abs(t.ppf(q / 2, f3))).quantize(Decimal('.0001')).__float__()

    print("\nПеревірка значимості коефіцієнтів регресії за критерієм Стьюдента: m = {}, N = {} ".format(m, n))
    averageVariation = numpy.average(list(map(numpy.var, y_table)))
    variationBetaS = averageVariation / n / m
    standardDeviationBetaS = math.sqrt(variationBetaS)
    Ti = [abs(betaCoeffs[i]) / standardDeviationBetaS for i in range(len(betaCoeffs))]
    f3 = (m - 1) * n
    q = 0.05
    ourT = getStudentVal(f3, q)
    importance = [True if el > ourT else False for el in list(Ti)]
    # print result data
    print("Оцінки коефіцієнтів βs: " + ", ".join(list(map(lambda x: str(round(float(x), 3)), betaCoeffs))))
    print("Коефіцієнти ts: " + ", ".join(list(map(lambda i: "{:.2f}".format(i), Ti))))
    print("f3 = {}; q = {}; tтабл = {}".format(f3, q, ourT))
    betaI = ["β0", "β1", "β2", "β3", "β12", "β13", "β23", "β123", "β11", "β22", "β33"]
    importanceToPrint = ["важливий" if i else "неважливий" for i in importance]
    toPrint = map(lambda x: x[0] + " " + x[1], zip(betaI, importanceToPrint))
    print(*toPrint, sep="; ")
    printEquation(betaCoeffs, importance)
    return importance


def fisherCriteria(m, N, d, tableX, tableY, coeffsB, importance):
    def getFisherVal(f3, f4, q):
        return Decimal(abs(f.isf(q, f4, f3))).quantize(Decimal('.0001')).__float__()

    f3 = (m - 1) * N
    f4 = N - d
    q = 0.05
    theorY = numpy.array([equationRegres(row[0], row[1], row[2], coeffsB) for row in tableX])
    avgY = numpy.array(list(map(lambda el: numpy.average(el), tableY)))
    Sad = m / (N - d) * sum((theorY - avgY) ** 2)
    variationsY = numpy.array(list(map(numpy.var, tableY)))
    Sv = numpy.average(variationsY)
    Fp = float(Sad / Sv)
    Ft = getFisherVal(f3, f4, q)
    theoreticalValsToPrint = list(
        zip(map(lambda x: "x1 = {0[1]:<10} x2 = {0[2]:<10} x3 = {0[3]:<10}".format(x), tableX), theorY))
    print("\nПеревірка адекватності моделі за критерієм Фішера: m = {}, N = {} для таблиці y_table".format(m, N))
    print("Теоретичні значення y для різних комбінацій факторів:")
    print("\n".join(["{arr[0]}: y = {arr[1]}".format(arr=el) for el in theoreticalValsToPrint]))
    print("Fp = {}, Ft = {}".format(Fp, Ft))
    print("Fp < Ft => модель адекватна" if Fp < Ft else "Fp > Ft => модель неадекватна")
    return True if Fp < Ft else False


def main(m, n):
    naturalPlan = generateFactorsTable(naturalPlanRaw)
    arrY = generateY(m, naturalPlanRaw)
    while not cochranCriteria(m, n, arrY):
        m += 1
        arrY = generateY(m, naturalPlan)

    printMatrix(m, n, naturalPlan, arrY, " для натуралізованих факторів:")
    coefficients = findCoeffs(naturalPlan, arrY)
    printEquation(coefficients)
    importance = studentCriteria(m, n, arrY, coefficients)
    d = len(list(filter(None, importance)))
    fisherCriteria(m, n, d, naturalPlan, arrY, coefficients, importance)


if __name__ == "__main__":
    m = 3
    n = 15
    main(m, n)
