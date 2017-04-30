import numpy as np
from numba import jit, int16
from numpy.linalg import cholesky, inv, det
import matplotlib.pyplot as plt
import pdb

"""Filtering implementation for state estimation
This module estimates the state of the UAV from given sensor meaurements
using unscented Kalman filter.
state:
    x: position, velocity
    R: attitude, angular velocity
    power: current, voltage, RPM
"""

def sigmaPoints(x, P, c):
    """Simg points computation"""
    A = c*cholesky(P).T
    Y = np.tile(x,((A.shape)[0],1)).T
    return np.vstack((x, Y+A, Y-A)).T

def dss(x):
    """State space UAV dynamics"""
    dt = 0.01
    A = np.zeros((12,12))
    for i in range(6):
        A[i][i], A[i][i+1] = 1, dt
        A[i+6][i+6] = 1
    return A.dot(x)

def sss(x):
    """Sensor state"""
    A = np.zeros((6,12))
    for i in range(6):
        A[i][i+6] = 1
    return A.dot(x)

def f(x):
    """nonlinear state function"""
    return np.array([x[1],x[2],0.05*x[0]*(x[1]+x[2])])

def h(x):
    """nonlinear sensor function"""
    return np.array([x[0]])

def ut(func, x, wm, wc, n_f, Q):
    """unscented transform
    args:
        f: nonlinear map
    output:
        mean
        sampling points
        covariance
        deviations
    """
    n, m = x.shape
    X = np.zeros((n_f,m))
    mu = np.zeros(n_f)
    for i in range(m):
        X[:,i] = func(x[:,i])
        mu = mu + wm[i]*X[:,i]
    Xd = X - np.tile(mu,(m,1)).T
    Sigma = Xd.dot(np.diag(wc.T).dot(Xd.T)) + Q
    return (mu, X, Sigma, Xd)

def ukf(x, P, z, Q, R, u):
    """UKF
    args:
        x: a priori state estimate
        P: a priori state covariance
        z: current measurement
        Q: process noise
        R: measurement noise
    output:
        x: a posteriori state estimation
        P: a posteriori state covariance
    """
    n = len(x)
    m = len(x)
    alpha = 0.75
    kappa = 0.
    beta = 2.
    lamb = alpha**2*(n+kappa)-n
    c_n = n+lamb
    Wm = np.append(lamb/c_n, 0.5/c_n+np.zeros(2*n))
    Wc = np.copy(Wm)
    Wc[0] +=  (1-alpha**2+beta)
    c_nr=np.sqrt(c_n)
    X = sigmaPoints(x,P,c_nr)
    x1, X1, P1, X2 = ut(dss, X, Wm, Wc, n, Q)
    z1,Z1,P2,Z2 =ut(sss, X1,Wm,Wc, int(n/2), R)
    P12=X2.dot(np.diag(Wc).dot(Z2.T))
    K=P12.dot(inv(P2))
    x=x1+K.dot(z-z1)
    P=P1-K.dot(P12.T)
    return x, P

@jit(int16(int16))
def test(a = 3):
    x = a
    return x

if __name__=='__main__':
    # test(4)
    # test.inspect_types()
    # print('test')
    Ns = 12 # number of states
    s = np.zeros(Ns)
    u = np.zeros(Ns)
    q=0.1
    r=0.1
    Q = q**2*np.eye(Ns)
    R = r**2
    P = np.eye(Ns)
    x = s+q*np.random.random(Ns)
    N = 20
    xV = np.zeros((Ns,N))
    sV = np.copy(xV)
    zV = np.zeros(N)
    for k in range(N):
        z = h(s)+r*np.random.random()
        sV[:,k] = s
        zV[k] = z
        x, P = ukf(x, P, z, Q, R, u)
        xV[:,k] = x
        s = dss(s) + q*np.random.random(Ns)
        pass
