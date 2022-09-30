import numpy as np
import math

tiesK_A = np.array([[511.80, 673.84], [511.80, 673.84]])

tiesK_B = np.array([[1.44, 632.52], [454.73, 645.03]])

tiesL_A = np.array([[508.93, 718.18], [508.93, 718.18]])

tiesL_B = np.array([[1.79, 684.58], [790.36, 860.60]])

print(tiesK_A.shape)

def angle_LoC (P1, P2, P3):
    a = np.linalg.norm(P2 - P1)
    b = np.linalg.norm(P3 - P1)
    c = np.linalg.norm(P3 - P2)
    return math.acos((a**2 + b**2 - c**2)/(2*a*b))

print(angle_LoC(tiesL_A[0], tiesK_A[0], tiesL_B[1]))



O = np.array([[2.13, 780, 22], [3.13, 300, 11]])

P = O / np.array([math.pi, O[:, 1].max(), O[:,2].max()])

print(P)





# M = np.array([[[1,2], [3,4], [5,6]], [[7,8], [9,10], [11,12]]])

# print(M[0])

# N = np.empty((11,2))

# #N[4] = M[0]

# print(N[4])

# print(M[0,1])

# N[4] = M[0,1]

# print(N[4])

# K = np.empty((2))

# K = M[1,2]

# print(K.shape)

# tiesK_A:
# [[511.79814291 673.8368187 ]
#  [511.79814291 673.8368187 ]]
# tiesK_B:
# [[  1.44052602 632.51695251]
#  [454.73059702 645.03516769]]
# tiesL_A:
# [[508.93065548 718.17666626]
#  [508.93065548 718.17666626]]
# tiesL_B:
# [[  1.78549895 684.57934189]
#  [790.36097717 860.59860229]]
