from sklearn.svm import SVC
import utils
import numpy as np


def gaussian_kernel(X, Y, sigma, device=None):
    D = utils.pairwise_distance(X, Y, device)
    ### YOUR CODE HERE ###
    return np.exp(-D/(2*sigma**2))
    
    


class svm(SVC):
    def __init__(self, kernel='gaussian', sigma=1, C=100, tol=.00001, device=None):
        if kernel == 'gaussian':
            self.sigma = sigma
            self.kernel = lambda X, Y: gaussian_kernel(X, Y, self.sigma, device)
        elif kernel == 'parity':
            self.kernel = parity_kernel
        elif kernel == 'linear':
            self.kernel = lambda X, Y: utils.linear_kernel(X, Y, device)
        else:
            self.kernel = kernel

        self.C = C
        self.tol = tol

    def decision_function(self, X):
        K = self.kernel(self.X, X)
        return (self.alpha*self.Y).dot(K) - self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def fit(self, X, Y, P=3):
        N, D = X.shape
        print("Number of training samples : %d"%N)
        self.K = self.kernel(X, X)
        self.alpha = np.zeros(N)
        self.b = 0

        KY = self.K * Y[None,:]
        p = 0
        while p < P:
            a = 0
            for i in range(N):
                yi = Y[i]
                Ei = self.alpha.dot(KY[i]) - self.b - yi
                if (yi*Ei < -self.tol and self.alpha[i] < self.C) \
                   or (yi*Ei > self.tol and self.alpha[i] > 0):
                    j = np.random.choice(N)
                    while j == i: j = np.random.choice(N)
                    yj = Y[j]
                    Ej = self.alpha.dot(KY[j]) - self.b - yj
                    if self.update_parameters(i, j, yi, yj, Ei, Ej):
                        a += 1
            if a == 0:
                p += 1
            else:
                p = 0

        I = self.alpha > self.tol
        self.X = X[I]
        self.alpha = self.alpha[I]
        self.Y = Y[I]
        self.K = self.K[I][:,I]
        print("Number of support vectors  : %d"%I.sum())
        print("Number of parameters       : %d"%(2*I.sum() + np.prod(self.X.shape)))

    def update_parameters(self, i, j, yi, yj, Ei, Ej):
        if yi == yj:
            L, H = max(0, self.alpha[i] + self.alpha[j] - self.C), min(self.C, self.alpha[i] + self.alpha[j])
        else:
            L, H = max(0, self.alpha[j] - self.alpha[i]), min(self.C, self.C + self.alpha[j] - self.alpha[i])

        if L == H: return False

        D = - self.K[j,j] + 2*self.K[i,j] - self.K[i,i]

        if D >= 0: return False

        ### YOUR CODE HERE ###
        # update alphaj and alphai
        # according to pseudo code
        alphaj = self.alpha[j] - (yj*(Ei - Ej)/D)
        
        if(alphaj > H):
            alphaj = H
        elif(alphaj < L):
            alphaj = L
        else:
            pass
        
        alphai = self.alpha[i] + yi*yj*(self.alpha[j] - alphaj)

        if abs(self.alpha[j] - alphaj) < 1e-10: return False

        b1 = self.b + Ei + yi*(alphai - self.alpha[i])*self.K[i,i] + yj*(alphaj - self.alpha[j])*self.K[i,j]
        b2 = self.b + Ej + yi*(alphai - self.alpha[i])*self.K[i,j] + yj*(alphaj - self.alpha[j])*self.K[j,j]
        self.alpha[i] = alphai
        self.alpha[j] = alphaj
        if self.alpha[i] > 0 and self.alpha[i] < self.C:
            self.b = b1
        elif self.alpha[j] > 0 and self.alpha[j] < self.C:
            self.b = b2
        else:
            self.b = (b1+b2)/2

        return True
