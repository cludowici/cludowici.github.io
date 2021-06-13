```python
import numpy as np
```

Something I've been trying to understand better over the pandemic has been the algorithms and mathematics that underpin commonly used statistical models. I've been reading a lot of great books. At the moment, I'm reading [Wood's book on Generalized Additive Models](https://www.routledge.com/Generalized-Additive-Models-An-Introduction-with-R-Second-Edition/Wood/p/book/9781498728331). He has a great chapter on mixed models. I see these models used a lot, but without much discussion of what's going on under the hood of whatever softward package computed the model. So in this post, I'll describe the logic of fitting these models with maximum likelihood estimation and write a function that does so in Python. 

The function I write here isn't meant to be a replacement for the complicated computing and estimation that underpins more specialized software for fitting these models. Instead, I hope it makes it clear how the mathematics translates into implementation. I avoid efficient, but sometimes hard to read, numerical computing tools like certain matrix decompositions in favor of readable code. 

# Linear Mixed Models

A simple mixed model has the form

$$Y = X\beta + Zb + \epsilon$$

Where b is the random effects, Z is a model matrix for the random effects, $\beta$ and X are the fixed effect coefficients and a design matrix, the same as a linear model, and $\epsilon$ is an error term

$$b \sim N(0, \psi_\theta)$$

$$\epsilon \sim N(0, \Lambda_\theta)$$

Thus 

$$y \sim N(X\beta, Z \psi_{\theta} Z^\intercal + \Lambda_\theta)$$

I write $\psi_\theta$ and $\Lambda_\theta$ because the target for our optimization procedure will be a set of parameters $\theta$ that these covariance matrices depend on.

### Maximum Likelihood Estimation for the model

We want to calculate estimates for $\beta$ and $\theta$.

Using $f()$ to denote density functions, a convenient expression for the model likelihood can be found as follows. 

1. Keep in mind that $f(y,b|\beta) = f(y|b,\beta)f(b)$
2. $f(y|\beta) = \int f(y,b|\beta)\, \mathrm{d}b$
3. Maximize $l(\beta, \theta) = \log f(y|\beta)$ to estimate $b$ and $\beta$

If we note that $y|b,\beta \sim N(X\beta+Zb, \Lambda_\theta)$ and $b \sim N(0, \psi_\theta)$, then:

$$f(y|\beta) = \int f(y,b|\beta)\, \mathrm{d}b = \int \exp[\log f(y,b|\beta)]\, \mathrm{d}b$$

The final expression can be expressed as a Taylor expansion about $\hat{b}$ - the maximizer of the log likelihood. We only need terms up to the second derivative, because the higher order derivatives of a log Gaussian are 0. The linear term in the expansion is 0 because $\hat{b}$ is a maximum.

$$
\int \exp[\,\log f(y,b|\beta)\,]\, \mathrm{d}b  = 
\int \exp[\,\log f(y,b|\beta) + 0 + (b-\hat{b})^{\intercal}\frac{1}{2} \frac{\partial^2\log f(y,b|\beta)}{\partial b \partial b^\intercal} (b-\hat{b})\,] \mathrm{d}b
\\= f(y,b|\beta)\int \exp[\,(b-\hat{b})^{\intercal}\frac{1}{2} \frac{\partial^2\log f(y,b|\beta)}{\partial b \partial b^\intercal} (b-\hat{b})\,] \mathrm{d}b
\\ = f(y,b|\beta)\int \exp[\,(b-\hat{b})^{\intercal}(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})(b-\hat{b})\,] \mathrm{d}b
$$ 

Now $\int \exp[\,(b-\hat{b})^{\intercal}(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})(b-\hat{b})\,] \mathrm{d}b$ is the kernel of a Gaussian with a mean of $\hat{b}$ and a covariance matrix $(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})^{-1}$. Because a probability distribution integrates to 1, this portion of the Gaussian integrates to the inverse of the normalization constant.

$$\int \exp[\,(b-\hat{b})^{\intercal}(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})(b-\hat{b})\,] \mathrm{d}b\\
= \frac{(2\pi)^{\frac{p}{2}}}{|(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})|^{\frac{1}{2}}}$$


So

$$f(y|\beta)\,=\,f(y,b|\beta)\int \exp[\,(b-\hat{b})^{\intercal}(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})(b-\hat{b})\,] \mathrm{d}b \\ 
= f(y,b|\beta)\,\frac{(2\pi)^{\frac{p}{2}}}{|(Z^{\intercal}\Lambda^{-1}_{\theta}Z\,+\,\psi^{-1}_{\theta})|^{\frac{1}{2}}}$$


From this, twice our log likelihood is:

$$2\,l(\beta, \theta) = -(y - X\beta - Z\hat{b})^\intercal\,\Lambda^{-1}_{\theta}(y - X\beta - Z\hat{b}) - \hat{b}^{\intercal}\psi^{-1}_{\theta}\hat{b} \\ 
-\log|\Lambda^{-1}_{\theta}| -\log|\psi^{-1}_{\theta}| - \log|Z^{\intercal}\Lambda^{-1}_{\theta}Z + \psi^{-1}_{\theta}| - n\log(2\pi)$$

Calculating the derivatives and solving for $\beta$ and $b$ reveals something that, at first, looks unsolveable, but in reality is just laborious. The MLEs for these parameters rely on each other. 

A MLE for $b$ is 

$$\hat{b} = \left(Z^{\intercal}\Lambda^{-1}_{\theta}Z + \psi^{-1}_{\theta}\right)^{-1} Z^{\intercal}\Lambda^{-1}_{\theta}(y - X\hat{\beta})$$

The MLE for $\beta$ is

$$\hat{\beta} = \left( X^\intercal \Lambda^{-1}_{\theta} X\right)^{-1}  \Lambda^{-1}_{\theta} X^\intercal \left(y - Zb\right)$$

One solution for  $\hat{\beta}$ is (from Henderson, 1950):

$$\hat{\beta} = 
    \left(
        X^\intercal \left\{\Lambda^{-1}_{\theta} - \Lambda^{-1}_{\theta}Z 
        \left[ Z^{\intercal} \Lambda^{-1}_{\theta}Z + \psi^{-1}_{\theta}\right]^{-1} Z^{\intercal} \Lambda^{-1}_{\theta} \right\} X\right)^{-1}
X^\intercal \left\{\Lambda^{-1}_{\theta} - \Lambda^{-1}_{\theta}Z 
        \left[ Z^{\intercal} \Lambda^{-1}_{\theta}Z + \psi^{-1}_{\theta}\right]^{-1} Z^{\intercal} \Lambda^{-1}_{\theta} \right\}y$$
        
So we now have a likelihood function that depends on $\beta$ and a way of estimating $b$ from $\beta$. Great! Let's try to implement this in NumPy.

## Doing the computations

I'll compute these estimators by maximum likelihood. The data are simple. An engineer wants to measure the time it takes certain ultrasonic waves to travel along a rail as a measure of longitudinal stress. There are three observations of the time taken for the wave for each of six rails. We're going to fit this with a model in which the intercept can vary between rails. This will give us a estimate of the variability between rails and the average time for each rail. More complex models have a more elaborate covariance matrix structure, which is beyond the scope of this post. I hope to write a post on that eventually.

Here's a function that computes the negative log likelihood for our model. Note the very simple covariance structure. There's no correlation between random effects here. 


```python
from scipy.stats import norm

def logLike(theta,X,Z,y, optimizing = True):
    
    sigmaRE = np.exp(theta[0]) #Ensure these are positive by inputing log values
    sigmaError = np.exp(theta[1])
  
    n = len(y)
    
    pr = Z.shape[1] #Dimensions of the random effects
    pf = X.shape[1] #Dimensions of the fixed effects

    inverse_lambda = np.eye(n) * 1/sigmaError #Very simple covariance matrices
    inverse_psi = np.eye(pr) * 1/sigmaRE

    yCovariance = Z.T @inverse_lambda @Z + inverse_psi

    #This term appears twice in the mixed model equation for beta_hat, compute it once
    C = (inverse_lambda - inverse_lambda @Z @ np.linalg.inv(yCovariance) @Z.T @inverse_lambda)
    
    #Compute \hat{\beta}
    firstTerm = np.linalg.inv(X.T @ C @X)

    secondTerm = X.T @ C @y

    beta_hat = firstTerm @ secondTerm #Solution to the mixed model equation

    #compute \hat{\b}
    b_hat = np.linalg.inv(yCovariance)@ Z.T @inverse_lambda @ (y - X @ beta_hat)

    #likelihood components
    logCovarDet = np.log(np.linalg.det( yCovariance ))

    sqrdMahalanobis  = (y - X @ beta_hat - Z @ b_hat).T @inverse_lambda @(y - X@beta_hat - Z @b_hat)

    lambda_ = np.eye(n) * sigmaError
    
    psi = np.eye(pr) * sigmaRE
    
    #log likelihood
    logLik = .5 *(- sqrdMahalanobis - \
    (b_hat.T @inverse_psi @b_hat) - \
    np.log(np.linalg.det(lambda_)) - \
    np.log(np.linalg.det(psi)) - \
    logCovarDet - \
    n*np.log(2*np.pi))

    if optimizing:
        return -logLik
    else:
        
        beta_hat_SD = np.sqrt(np.linalg.inv(X.T @ np.linalg.inv(Z@psi@Z.T + lambda_)@X))
        p_value = 2*(1- norm.cdf(beta_hat,loc = 0, scale =  beta_hat_SD))
        z = beta_hat/beta_hat_SD
        
        return {'Log Likelihood': logLik, 'Parameters': np.array(np.r_[beta_hat, b_hat]), 'Beta_SD':beta_hat_SD, 'z':z, 'p-value': p_value}
```

Where does that p-value come from? Remember that $y \sim N(X\beta, Z \psi_{\theta} Z^\intercal + \Lambda_\theta)$. If we calculate the the MLE for $\beta$ from this, we have.

$$\hat{\beta} = \left\{X^\intercal \left( Z\psi Z^\intercal + \Lambda \right) ^{-1} X \right\}^{-1} X^\intercal \left( Z\psi Z^\intercal + \Lambda \right) ^{-1}y$$


$$\mathbb{E} \left[ \hat{\beta} \right] = \left\{X^\intercal \left( Z\psi Z^\intercal + \Lambda \right) ^{-1} X \right\}^{-1} X^\intercal \left( Z\psi Z^\intercal + \Lambda \right) ^{-1} \mathbb{E} \left[y  \right] \\
= \left\{X^\intercal \left( Z\psi Z^\intercal + \Lambda \right) ^{-1} X \right\}^{-1} X^\intercal \left( Z\psi Z^\intercal + \Lambda \right) ^{-1} X\beta \\
= \beta$$


$$\mathbb{Var} \left[ \hat{\beta} \right] =  \left\{X^\intercal \left( Z\psi Z^\intercal + \Lambda \right) ^{-1} X \right\}^{-1} X^\intercal \left( Z\psi Z^\intercal + \Lambda \right) ^{-1}
\left( Z\psi Z^\intercal + \Lambda \right) 
\left( Z\psi Z^\intercal + \Lambda \right) ^{-1} X
\left\{X^\intercal \left( Z\psi Z^\intercal + \Lambda \right) ^{-1}X\right\}^{-1}\\
= \left\{X^\intercal \left( Z\psi Z^\intercal + \Lambda \right) ^{-1} X \right\}^{-1} \left[X^\intercal
\left( Z\psi Z^\intercal + \Lambda \right) ^{-1} X\right]
\left\{X^\intercal \left( Z\psi Z^\intercal + \Lambda \right) ^{-1}X\right\}^{-1}\\
= \left\{X^\intercal \left( Z\psi Z^\intercal + \Lambda \right) ^{-1}X\right\}^{-1}$$


Because $\hat{\beta}$ is a linear transform of y, which has a Gaussian distribution, it also has a Gaussian distribution.

Great. Now I'll read in the data. The first column is the rail labels, the second is the travel time for the wave.


```python
railData = np.genfromtxt('Rail.csv', delimiter =  ',', skip_header= 1, usecols = (1,2))

railData
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



Here's the random effects matrix $Z$. All we care about for this model is separate intercepts for each rail.


```python
Z = np.zeros((railData.shape[0],6)) #Random effects matrix
for col in range(6):
    Z[col*3:col*3+3, col] = [1,1,1]
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



The fixed effect $X$ only contains an intercept.


```python
X = np.ones((railData.shape[0],1)) #Fixed effects
```

$y$ is the travel time for the waves.


```python
y = railData[:,1]
y
```




    array([ 55.,  53.,  54.,  26.,  37.,  32.,  78.,  91.,  85.,  92., 100.,
            96.,  49.,  51.,  50.,  80.,  85.,  83.])



Here we minimize the negative log likelihood.


```python
from scipy.optimize import minimize
out = minimize(logLike, [0,0], (X, Z, y))
```

I've written the function so that if we plug in our variance estimates and tell the function we're not optimizing, it will return our random effects, fixed intercept and a p-value for the fixed effect.


```python
results = logLike(out.x, X, Z, y, optimizing = False)
results
```




    {'Log Likelihood': -64.2800184692185,
     'Parameters': array([ 66.5       , -12.3697708 , -34.47042796,  17.97740023,
             29.19265909, -16.32809746,  15.9982369 ]),
     'Beta_SD': array([[9.28484835]]),
     'z': array([[7.16220637]]),
     'p-value': array([[7.93809463e-13]])}



Let's compare it to the output of the statsmodels mixed model function.


```python
import statsmodels.api as sm

import statsmodels.formula.api as smf

import pandas as pd

data = pd.DataFrame(railData, columns = ('Rail', 'Travel'))

statsModelLMM = smf.mixedlm("Travel ~ 1", data, groups = data['Rail']).fit(method = 'BFGS', reml = False)

statsModelLMM.summary()
```




<table class="simpletable">
<tr>
       <td>Model:</td>       <td>MixedLM</td> <td>Dependent Variable:</td>  <td>Travel</td> 
</tr>
<tr>
  <td>No. Observations:</td>   <td>18</td>          <td>Method:</td>          <td>ML</td>   
</tr>
<tr>
     <td>No. Groups:</td>       <td>6</td>          <td>Scale:</td>         <td>16.1668</td>
</tr>
<tr>
  <td>Min. group size:</td>     <td>3</td>      <td>Log-Likelihood:</td>   <td>-64.2800</td>
</tr>
<tr>
  <td>Max. group size:</td>     <td>3</td>        <td>Converged:</td>         <td>Yes</td>  
</tr>
<tr>
  <td>Mean group size:</td>    <td>3.0</td>            <td></td>               <td></td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>       <th>Coef.</th>  <th>Std.Err.</th>   <th>z</th>   <th>P>|z|</th> <th>[0.025</th> <th>0.975]</th>
</tr>
<tr>
  <th>Intercept</th> <td>66.500</td>    <td>9.285</td>  <td>7.162</td> <td>0.000</td> <td>48.302</td> <td>84.698</td>
</tr>
<tr>
  <th>Group Var</th> <td>511.851</td>  <td>90.961</td>    <td></td>      <td></td>       <td></td>       <td></td>   
</tr>
</table>



These are the same models! Our estimates of the variance of the random effects and the residual variance are


```python
np.round(np.exp(out.x),2)
```

The estimates from the statsmodel model are


```python
np.round([statsModelLMM.cov_re['Group'][0], statsModelLMM.scale],2)
```

Our fixed intercept and the statsmodel fixed intercept


```python
np.round([results['Parameters'][0], statsModelLMM.fe_params['Intercept']],2)
```

Do our random intercepts and the statsmodel random intercepts agree?


```python
np.allclose(
    np.array([value[1] for value in statsModelLMM.random_effects.items()]).flatten(),
    results['Parameters'][1:]
)
```




    True


