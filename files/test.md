```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
railData = np.genfromtxt('Rail.csv', delimiter =  ',', skip_header= 1, usecols = (1,2))
```




    array([[  1.,  55.],
           [  1.,  53.],
           [  1.,  54.],
           [  2.,  26.],
           [  2.,  37.],
           [  2.,  32.],
           [  3.,  78.],
           [  3.,  91.],
           [  3.,  85.],
           [  4.,  92.],
           [  4., 100.],
           [  4.,  96.],
           [  5.,  49.],
           [  5.,  51.],
           [  5.,  50.],
           [  6.,  80.],
           [  6.,  85.],
           [  6.,  83.]])



# Linear Mixed Models

A simple mixed model has the form

$$Y = X\beta + Zb + \epsilon$$

Where b is the random effects, Z is a set of coefficients for those effects (a model matrix), X and $\beta$ are fixed effect coefficients and observations, and $\epsilon$ is an error term

$$b \sim N(0, \psi_\theta)$$

$$\epsilon \sim N(0, \Lambda_\theta)$$

Thus 

$$y \sim N(X\beta, Z \psi_{\theta} Z^\intercal + \Lambda_\theta)$$

## Maximum Likelihood Estimation for the model

We want to calculate estimates for $\beta$ and $\theta$.

We could calculate the model likelihood from the distribution of y, $N(X\beta, Z \psi_{\theta} Z^\intercal + \Lambda_\theta)$, but we need $(Z \psi_{\theta} Z^\intercal + \Lambda_\theta)^{-1}$ for this, which is computationally demanding because it is $n \times n$ and thus the number of operations scales as $O(n^3)$.

A more convenient expression can be found as follows.

1. Find $f(y|b,\beta)$
2. Find $f(b)$
3. Calculate $f(y,b|\beta) = f(y|b,\beta)f(b)$
4. $f(y|\beta) = \int\!f(y,b|\beta)\, \mathrm{d}b$

If we note that $y|b,\beta \sim N(X\beta+Zb, \Lambda_\theta)$ and $b \sim N(0, \psi_\theta)$, then:

$$f(y|\beta) = \int\!f(y,b|\beta)\, \mathrm{d}b = \int\!\exp[\log\!f(y,b|\beta)]\, \mathrm{d}b$$

The final expression can be expressed as a Taylor expansion about $\hat{b}$ - the maximizer of the log likelihood. We only need terms up to the second derivative, because the higher order derivatives of a log Gaussian are 0. The linear term in the expansion is 0 because $\hat{b}$ is a maximum.

$$
\int\!\exp[\,\log\!f(y,b|\beta)\,]\, \mathrm{d}b  = 
\int\!\exp[\,\log\!f(y,b|\beta) + 0 + (b-\hat{b})^{\intercal}\frac{1}{2} \frac{\partial^2\log\!f(y,b|\beta)}{\partial b \partial b^\intercal} (b-\hat{b})\,] \mathrm{d}b
\\= f(y,b|\beta)\int\!\exp[\,(b-\hat{b})^{\intercal}\frac{1}{2} \frac{\partial^2\log\!f(y,b|\beta)}{\partial b \partial b^\intercal} (b-\hat{b})\,] \mathrm{d}b
\\ = f(y,b|\beta)\int\!\exp[\,(b-\hat{b})^{\intercal}(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})(b-\hat{b})\,] \mathrm{d}b
$$ 

Now $\int\!\exp[\,(b-\hat{b})^{\intercal}(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})(b-\hat{b})\,] \mathrm{d}b$ is the kernel of a Gaussian with a mean of $\hat{b}$ and a covariance matrix $(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})^{-1}$. Because a probability distribution integrates to 1, this portion of the Gaussian integrates to the inverse of the normalization constant.

$$\int\!\exp[\,(b-\hat{b})^{\intercal}(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})(b-\hat{b})\,] \mathrm{d}b\\
= \frac{(2\pi)^{\frac{p}{2}}}{|(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})|^{\frac{1}{2}}}$$


So

$$f(y|\beta)\,=\,f(y,b|\beta)\int\!\exp[\,(b-\hat{b})^{\intercal}(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})(b-\hat{b})\,] \mathrm{d}b \\ 
= f(y,b|\beta)\,\frac{(2\pi)^{\frac{p}{2}}}{|(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})|^{\frac{1}{2}}}$$


From this, twice our log likelihood is:

$$2\,l(\beta, \theta) = -(y - X\beta - Z\hat{b})^\intercal\,\Lambda^{-1}_{\theta}(y - X\beta - Z\hat{b}) - \hat{b}^{\intercal}\psi^{-1}_{\theta}\hat{b} \\ 
-\log|\Lambda^{-1}_{\theta}| -\log|\psi^{-1}_{\theta}| - \log|Z^{\intercal}\Lambda^{-1}_{\theta}Z + \psi^{-1}_{\theta}| - n\log(2\pi)$$

Calculating the derivatives and solving for $\beta$ and $b$ reveals something that, at first, looks unsolveable. The MLEs for these parameters rely on each other. 

A MLE for $b$ is 

$$\hat{b} = \left(Z^{\intercal}\Lambda^{-1}_{\theta}Z + \psi^{-1}_{\theta}\right)^{-1} Z^{\intercal}\Lambda^{-1}_{\theta}(y - X\hat{\beta})$$

The MLE for $\beta$ is

$$\hat{\beta} = ...$$

However, there's a useful - but laborious - result for estimating $\beta$:

$$\hat{\beta} = \left(X^\intercal \left\{\Lambda^{-1}_{\theta} - \Lambda^{-1}_{\theta}Z \left[ Z^{\intercal} \Lambda^{-1}_{\theta}Z + \psi^{-1}_{\theta} ]^{-1} Z^{\intercal} \Lambda^{-1}_{\theta}\right] \right\} X\right)^{-1}
\left(\Lambda^{-1}_{\theta} - \Lambda^{-1}_{\theta}Z \left[ Z^{\intercal} \Lambda^{-1}_{\theta}Z + \psi^{-1}_{\theta} ]^{-1} Z^{\intercal} \Lambda^{-1}_{\theta}\right] \right)y$$

This is apparently due to Henderson (1950), but I can't find a PDF to verify it. 

### Other distributions

I'll leave the derivations out for these. They follow pretty readily from standard results for Gaussians.

$$\hat{\beta}\,\sim\,N(\beta, [\,X^\intercal (Z Z^\intercal + \Lambda)^{-1}X\,]^{-1})$$


$$b|y,\hat{\beta}\,\sim\,N(\hat{b}, (Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})^{-1})$$


Hang on, doesn't $(Z \psi Z^\intercal + \Lambda)^{-1}$ mean we'll have to estimate and compute that costly covariance matrix inversion that we were trying to avoid? Well, if we treat $\beta$ as a vector of random effects with uniform priors, then we can partition $f(b|y,\beta)$ and find a computationally less demanding ($O(np^2)$ rather than $O(n^3)$) solution to the covariance matrix inverse (see Wood for details)

Finally, we use the large sample limit of the maximum likelihood estimates for inference about those estimates

$$\hat{\theta}\,\sim\,N(\theta, \hat{\mathcal{I}}_{p})$$

Where $\hat{\mathcal{I}}_{p}$ is an estimate of Fisher information $$\hat{\mathcal{I}}_{p} = -\left[ \frac{\partial^2 \log\,l(\theta, \hat{\beta})}{\partial\theta^2} \right]$$


```python

def llm(theta,X,Z,y, returnEstimates = False):
    
    sigmaRE = np.exp(theta[0])
    sigmaError = np.exp(theta[1])
  
    n = len(y)
    
    pr = Z.shape[1] #Dimensions of the random effects
    pf = X.shape[1] #Dimensions of the fixed effects

    inverse_lambda = np.eye(n) * 1/sigmaError #Very simple covariance matrices
    inverse_psi = np.eye(pr) * 1/sigmaRE

    yCovariance = Z.T @inverse_lambda @Z + inverse_psi

    #This term appears twice in the mixed model equation for beta_hat, compute it once
    C = (inverse_lambda - inverse_lambda @Z @ np.linalg.inv(yCovariance) @Z.T @inverse_lambda)

    firstTerm = np.linalg.inv(X.T @ C @X)

    secondTerm = X.T @ C @y

    beta_hat = firstTerm @ secondTerm #Solution to the mixed model equation

    b_hat = np.linalg.inv(yCovariance)@ Z.T @inverse_lambda @ (y - X @ beta_hat)

    #Profile likelihood components
    logCovarDet = np.log(np.linalg.det( yCovariance ))

    mahalanobis = (y - X @ beta_hat - Z @ b_hat).T @inverse_lambda @(y - X@beta_hat - Z @b_hat)

    logLik = .5 *(- mahalanobis - \
    (b_hat.T @inverse_psi @b_hat) - \
    np.log(np.linalg.det(np.linalg.inv(inverse_lambda))) - \
    np.log(np.linalg.det(np.linalg.inv(inverse_psi))) - \
    logCovarDet - \
    n*np.log(2*np.pi))

    if not returnEstimates:
        return -logLik
    else:
        print(b_hat)
        return {'Likelihood': np.exp(logLik), 'Parameters': np.array(np.r_[beta_hat, b_hat])}
```


```python
Z = np.zeros((18,6)) #Random effects matrix
Z[0:3,0] = 1
Z[3:6,1] = 1
Z[6:9,2] = 1
Z[9:12,3] = 1
Z[12:15,4] = 1
Z[15:18,5] = 1
Z
```




    array([[1., 0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0., 1.]])




```python
X = np.ones((18,1)) #Fixed effects
y = railData[:,1]

y
```




    array([ 55.,  53.,  54.,  26.,  37.,  32.,  78.,  91.,  85.,  92., 100.,
            96.,  49.,  51.,  50.,  80.,  85.,  83.])




```python
from scipy.optimize import minimize


out = minimize(llm, [0,0], (X, Z, y))

out
```




          fun: 64.2800184692185
     hess_inv: array([[ 0.34495004, -0.0011941 ],
           [-0.0011941 ,  0.16654592]])
          jac: array([2.86102295e-06, 9.53674316e-07])
      message: 'Optimization terminated successfully.'
         nfev: 69
          nit: 20
         njev: 23
       status: 0
      success: True
            x: array([6.23805421, 2.78295153])




```python
llm(out.x, X, Z, y, returnEstimates=True)
```

    [-12.36977069 -34.47042767  17.97740008  29.19265884 -16.32809732
      15.99823676]





    {'Likelihood': 1.4692150362641053e-56,
     'Parameters': array([ 66.5       , -12.36977069, -34.47042767,  17.97740008,
             29.19265884, -16.32809732,  15.99823676])}




```python
import statsmodels.api as sm

import statsmodels.formula.api as smf

import pandas as pd

data = pd.DataFrame(railData, columns = ('Rail', 'Travel'))

statsModelLMM = smf.mixedlm("Travel ~ 1", data, groups = data['Rail']).fit(method = 'BFGS', reml = False)

statsModelLMM.llf, statsModelLMM.random_effects, statsModelLMM.conf_int(.05)
```




    (-64.28001847009503,
     {1.0: Group   -12.369767
      dtype: float64,
      2.0: Group   -34.470417
      dtype: float64,
      3.0: Group    17.977394
      dtype: float64,
      4.0: Group    29.19265
      dtype: float64,
      5.0: Group   -16.328092
      dtype: float64,
      6.0: Group    15.998232
      dtype: float64},
                        0          1
     Intercept  48.302219  84.697781
     Group Var -12.679025  76.000153)




```python

```
