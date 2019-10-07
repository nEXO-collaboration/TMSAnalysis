import numpy as np
from scipy.special import erf

###########################################################################
def TwoExpConv(x,A,mu,t1,t2):

    # Convolution of two exponentials

    mask = (x-mu)<0
    y = A/(t1-t2)*( np.exp(-(x-mu)/t1) - np.exp(-(x-mu)/t2) )
    y[mask] = np.zeros(len(y))[mask]
    return y


###########################################################################
def GumbelDist(x,A,mu,beta):

    # Gumbel distribution

    scale_factor = 0.367879376 # so that if A = 1, the function peaks at 1.
    return A/scale_factor * np.exp( -(x-mu)/beta - np.exp( -(x-mu)/beta ) )


###########################################################################
def DoubleExpGaussConv(x,A,B,mu,sig,t1,t2): 

    # Convolution of a sum-of-two-exponentials and a gaussian.

    x = x - mu
    y = A * \
        ( B/(2.*t1)*np.exp(sig**2/(2.*t1**2)) * np.exp(-x/t1) * \
          (1. + erf( (x-sig**2./t1)/(sig*np.sqrt(2.)) )) + \
          (1.-B)/(2.*t2)*np.exp(sig**2/(2.*t2**2)) * np.exp(-x/t2) * \
          (1. + erf( (x-sig**2./t2)/(sig*np.sqrt(2.)) )) \
        )
    return y

###########################################################################
def Gaussian( x, A, mu, sig ):
    return A*np.exp(-(x-mu)**2/(2 * sig**2))


###########################################################################
def fitShaped( x, p0, p1, p2, p3, p4 ):
    fenzi = np.exp( -(x-p1)/p2 ) - np.exp( -(x-p1)/p3 )
    fenmu = p2/(p2-p3)
    result = p0 * fenzi/fenmu + p4
    result[x<p1] = np.ones(len(result[x<p1]))*p4
    return result

###########################################################################
def fitShapedConst( x, p0, p1, p4 ):
    p2 = 125.
    p3 = 62.5
    fenzi = np.exp( -(x-p1)/p2 ) - np.exp( -(x-p1)/p3 )
    fenmu = p2/(p2-p3)
    result = p0 * fenzi/fenmu + p4
    result[x<p1] = np.ones(len(result[x<p1]))*p4
    return result

###########################################################################
def DoubleExpGaussConvTFixed(x,A,B,mu,sig):
    return DoubleExpGaussConv(x,A,B,mu,sig,0.918,14.2)

