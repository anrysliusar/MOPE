import numpy as np


def getMatrix(s, r):
    return np.array([[np.random.randint(1, 21) for i in range(s)] for j in range(r)])


def getRandomNum(minValue, maxValue, countValue):
    return [np.random.randint(minValue, maxValue) for i in range(countValue)]


def finderY(a0, a1, a2, a3, X1, X2, X3):
    return a0 + a1 * X1 + a2 * X2 + a3 * X3


def finderXn(x, x0, dx):
    return (x - x0) / dx


def showMatrix(matrix):
    for i in matrix:
        print(i)
    print('-' * 50)


m = getMatrix(3, 8)
[a0, a1, a2, a3] = getRandomNum(1, 15, 4)

# calculating listY
listY = []
for i in range(len(m)):
    [X1, X2, X3] = m[i]
    listY.append(finderY(a0, a1, a2, a3, X1, X2, X3))

# calculating listX0, listDx
listX0 = []
listDx = []
mTransp = m.transpose()

for col in mTransp:
    maxXi = max(col)
    minXi = min(col)
    X0 = (maxXi + minXi) / 2
    listX0.append(X0)
    listDx.append(maxXi - X0)

# calculating matrixXn
matrixXn = []
num = 0
for col in mTransp:
    tempListXn = []
    for x in col:
        tempListXn.append(finderXn(x, listX0[num], listDx[num]))
    matrixXn.append(tempListXn)
    num += 1
arrayXn = np.array(matrixXn).transpose()


# calculating Yэт
[X01, X02, X03] = listX0
Yet = finderY(a0, a1, a2, a3, X01, X02, X03)

# calculating list Of Optimality Criterion
listOptimalityCriterion = []
for y in listY:
    listOptimalityCriterion.append(pow((y - Yet), 2))

showMatrix(m)
print(listY)
print('-' * 50)
showMatrix(arrayXn)
print(arrayXn[listOptimalityCriterion.index(max(listOptimalityCriterion))])
print(f"max((Y - Yэт)^2) = {max(listOptimalityCriterion)}")
