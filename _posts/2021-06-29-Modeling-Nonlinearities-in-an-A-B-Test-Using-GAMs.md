<link href="css.css" rel="stylesheet"></link>
It's often the case in an A/B test that covariates are added to a model in order to reduce variance, improve the precision of estimates, or look for conditional effects. However, often this either relies on the assumption that the covariate effects are linear, or uses unwieldy basis expansions like polynomials to account for nonlinear relationships. In this post I show how to use generalized additive models (GAMs) to account for nonlinearities in the relationships between covariates and outcome measures. I use data from an randomized A/B test that looked for differences in profit between two groups of businesses. The effect is small and there's a lot of variance in the data, but this sort of messy data is exactly what we often see in applications, and it's exactly where the easy gains in variance explained by nonlinearities are most useful.

## Generalized Additive Models and Smoothing Splines

GAMs are additive models of the form

$$g(\mathbb{E}\left[ y\right]) = \alpha + \sum_j f_j(x_j)$$

The function $g$ is a link function for an exponential family distribution that acts in the same way as link functions in generalized linear models. In this post I'll use a Gaussian distribution and an identity link for ease of explanation. The theory for response distributions and the associated link functions is the same as that for GLMs.

The $f_j$ are smooth functions of the j-th covariate. These are estimated using interpolation methods that are *penalized* to attenuate overfitting. Often they are piecewise functions of the data with some continuity constraints, like a cublic spline.  They minimize the difference between the outcome $y$ and the function $f$ with a penalty $\lambda$ on the "wiggliness" of the function.

$$\mid\mid y - f \mid\mid^2 +\,\lambda\int \left(\frac{\partial^2 f}{\partial x^2}\right)^2dx$$

The greater $\lambda$ is, the less "wiggly" the function is allowed to be. In the unpenalized scenario ($\lambda = 0$), the function interpolates all points, which is obviously undesirable because it leads to massive overfitting. In the completely penalized scenario ($\lambda \to \infty$), the fitting cannot accept any nonzero second derivatives and the function is linear. 

I use `mgcv::` in R to estimate the parameters for GAMs here. It uses generalized cross validation to estimate $\lambda$ and combines this with iterative reweighted least squares (i.e. Newton's method) for optimization. 

I highly, highly reccommend [Simon Wood's GAM book](https://www.routledge.com/Generalized-Additive-Models-An-Introduction-with-R-Second-Edition/Wood/p/book/9781498728331) for a comprehensive coverage of GAM theory and use. 

## The Data

The data are a set of clustered observations taken from an RCT described by  ([Atkin, Khandelwal, & Osman, 2017](https://academic.oup.com/qje/article/132/2/551/3002609)). The authors had a stratified sample of Egyptian rug manufacturers randomized to either receive opportunities to export their rugs to foreign markets, or for no offer to be made. This tests the hypothesis that exposure to foreign markets can improve profits. The manufacturers' monthly profits were recorded over 16 to 24 months, resulting in clustering within manufacturers, along with a number of measures of rug quality (i.e. thread count) and some demographic information which we'll ignore here. The data are made available by The Abdul Latif Jameel Poverty Action Lab and are [hosted on the Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QOGMVI). 

First up, I want to double check that I have the right data. The simplest way to do this is to calculate the counts for various subsets of the data and then compare with the published record in Atkin, Khandelwal, and Osman (2017). To do this I'll need to merge some data, because there are some observations in the main dataset that lack information about treatment takeup etc.


```R
# Sat Jun 26 13:37:09 2021 ------------------------------
library(ggplot2)
library(mgcv)
library(nlme)
library(magrittr)
library(dplyr)
library(kableExtra)
library(IRdisplay)

theme_apa <- papaja::theme_apa() + 
    theme(text = element_text(size = 20)) + 
    theme(plot.title = element_text(hjust = 0.5))

options(repr.plot.width = 20, repr.plot.height = 10)

displayAsHTML <- function(x)
    display_html(as.character(kable(x)))

main <- haven::read_dta('JPAL_3813/data/analysis.dta') #Main dataset, for analysis

takeupData <- haven::read_dta('JPAL_3813/data/takeup.dta') #Some missing data can be found here. I need to match on ID and round and select 
```

What I need to do is match the datasets on manufacturer ID (`main$id`) and the time of the observation (`main$round`), select the missing observations and place them in `main`. `dplyr::anti_join` can do this in a way that's similar to combining a `LEFT JOIN` in SQL with some filtering of matches.


```R
#Here, use anti_join to find the rows in takeupData with no matching ids and rounds in main
missingFromMain = anti_join(takeupData, main, c('id', 'round'))

#Empty tibble with all the columns that we want
newRows = nrow(missingFromMain)
empties = main[2352:(2351+newRows),]

#Plug in some missing data
empties$id = missingFromMain$id
empties$round = missingFromMain$round
empties$produced = missingFromMain$produced
empties$takeup = missingFromMain$takeup

#Loop over the newrows to fill in some details for aggregating. Etype is rug type
for(row in 1:newRows){
  thisID = empties$id[row]
  strata = main$strata[main$id == thisID][1]
  treatment = main$treatment[main$id == thisID][1]
  etype = main$etype[main$id == thisID][1]
  
  empties$strata[row] = strata
  empties$treatment[row] = treatment
  empties$etype[row] = etype
}

main = rbind(main, empties)
```


```R
main %<>% #Rename the etype for ease of interpretation
 mutate(etype = replace(etype, etype == 1, 'Duble (sample 1)')) %>%
 mutate(etype = replace(etype, etype == 2, 'Tups')) %>%
 mutate(etype = replace(etype, etype == 3, 'Goublan')) %>%
 mutate(etype = replace(etype, etype == 4, 'Kasaees')) %>%
 mutate(etype = replace(etype, etype == 5, 'Duble (sample 2)')) %>%
 mutate(etype = factor(etype)) %>%
 rename(rugType = etype)
```


```R
main %>%
filter(!is.na(rugType) & etype != 0) %>%
group_by(rugType) %>%
summarise(N = length(unique(id)))
```


<table class="dataframe">
<caption>A tibble: 6 × 2</caption>
<thead>
	<tr><th scope=col>rugType</th><th scope=col>N</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>0               </td><td> 17</td></tr>
	<tr><td>Duble (sample 1)</td><td> 79</td></tr>
	<tr><td>Duble (sample 2)</td><td>140</td></tr>
	<tr><td>Goublan         </td><td>103</td></tr>
	<tr><td>Kasaees         </td><td> 38</td></tr>
	<tr><td>Tups            </td><td> 83</td></tr>
</tbody>
</table>




```R
#There are multiple observations for each unit
main %>%
filter(!is.na(rugType) & rugType != 0) %>%
group_by(id, rugType) %>%
summarise(treatIndicator = sum(treatment)/length(treatment), .groups='drop')%>% #This returns a indicator for each unit being treated
group_by(rugType) %>%
summarise('Num. Treated' = sum(treatIndicator), 'Num. Control' = sum(treatIndicator==0))
```


<table class="dataframe">
<caption>A tibble: 5 × 3</caption>
<thead>
	<tr><th scope=col>rugType</th><th scope=col>Num. Treated</th><th scope=col>Num. Control</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Duble (sample 1)</td><td>39</td><td> 40</td></tr>
	<tr><td>Duble (sample 2)</td><td>35</td><td>105</td></tr>
	<tr><td>Goublan         </td><td>49</td><td> 54</td></tr>
	<tr><td>Kasaees         </td><td>19</td><td> 19</td></tr>
	<tr><td>Tups            </td><td>42</td><td> 41</td></tr>
</tbody>
</table>




```R
#table 1 row 6
main[main$round == 203 & main$takeup == 1,] %>% 
filter(!is.na(rugType) & etype != 0) %>%
group_by(rugType) %>%
summarise('Num Takeup Firms' = length(unique(id)), 'mean|takeup' = round(mean(produced)), 'sd|takeup' = round(sd(produced)))
```


<table class="dataframe">
<caption>A tibble: 4 × 4</caption>
<thead>
	<tr><th scope=col>rugType</th><th scope=col>Num Takeup Firms</th><th scope=col>mean|takeup</th><th scope=col>sd|takeup</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Duble (sample 1)</td><td>14</td><td>778</td><td>333</td></tr>
	<tr><td>Duble (sample 2)</td><td>32</td><td>434</td><td>178</td></tr>
	<tr><td>Goublan         </td><td> 5</td><td>586</td><td>368</td></tr>
	<tr><td>Tups            </td><td> 8</td><td>589</td><td>423</td></tr>
</tbody>
</table>



These are all the same as the published counts. However, as can be seen in the last table, the experimenters were only able to secure foreign buyers for one kind of rug, known as a "duble". I'll restrict my analyses to this kind of rug and ignore the first set of profit observations so that I have a baseline to include in the modeling. I'll do pairwise deletions for missing data, a choice that the authors also made.


```R
#Indicator Vars
dubs_sample = (main$strata %in% c(1,2,3,4,21,22,23,24)) & (!main$round %in% c(100,200)) & #Select duble strata and ignore first set of obs.
    complete.cases(main[,c('tp', 'strata','round', 'log_profit_rug_business_b', 'log_profit_rug_business', 'id')])
```

## Modeling

Now we'll start to build GAMs to test for differences between treatment and control. Notice in the tables above that among the duble firms, not every group exposed to treatment (i.e. offered a chance to export internationally) took the experimenters up on the offer. I'm going to include data from the firms that didn't take up the offer in my models. This is an 'intent to treat' design, which attempts to avoid the effects of non-random attrition on the treatment randomization. 


### Checking Model Assumptions

We're going to model log profit data for each month with a series of predictors in order to estimate our treatment effect. Let's have a look at the distribution of the log profit data. 


```R
library(ggplot2)

main %<>% mutate(log_profit_rug_business_b = log_profit_rug_business_b - mean(log_profit_rug_business_b, na.rm=TRUE)) #center

ggplot(main[dubs_sample,])+
    geom_histogram(aes(x = log_profit_rug_business))+
    labs(x = 'log(profit)')+
    theme_apa


ggplot(main[dubs_sample,])+
    stat_qq(aes(sample = log_profit_rug_business))+
    stat_qq_line(aes(sample = log_profit_rug_business))+
    labs(title = 'log(profit) Normal QQ')+
    theme_apa
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    



    
![png](/Figures/2021-06-29/output_14_1.png)
    



    
![png](/Figures/2021-06-29/output_14_2.png)
    


GAMs are generalized, in the style of GLMs, so we could use any exponential family distribution we wanted too, with an appropriate link function. However, a Gaussian with an identity link would work well here. There are some deviations from Gaussian quantiles in the QQ plot, but nothing too dramatic.  `Mgcv::` defaults to the Gaussian, so I don't need to call the model with any special arguments to use it.

## Modelling with `mgcv::`

### Model 1: Treatment only

A model that ignores the structure of the experiment entirely would regress log profit only on the treatment effect. This, thanks to randomization, gives an unbiased estimate of the treatment effect, but ignores a lot of information we have about things like clustering and the time relationship between the observations.


```R
#Modelling
modTreatNoCovar <- gam(log_profit_rug_business ~ treatment, data = main[dubs_sample,])

summary(modTreatNoCovar)
```


    
    Family: gaussian 
    Link function: identity 
    
    Formula:
    log_profit_rug_business ~ treatment
    
    Parametric coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  6.64437    0.02852 232.993  < 2e-16 ***
    treatment    0.21799    0.04803   4.539 6.91e-06 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    
    R-sq.(adj) =  0.0331   Deviance explained = 3.48%
    GCV = 0.30277  Scale est. = 0.30171   n = 573


Of course, running this using `mgcv::gam()` is overkill. Because without smoothed variables or random effects, this is a normal linear model.


```R
summary(lm(log_profit_rug_business ~ treatment, data = main[dubs_sample,]))
```


    
    Call:
    lm(formula = log_profit_rug_business ~ treatment, data = main[dubs_sample, 
        ])
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -2.41184 -0.31290  0.07999  0.27926  1.72387 
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  6.64437    0.02852 232.993  < 2e-16 ***
    treatment    0.21799    0.04803   4.539 6.91e-06 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 0.5493 on 571 degrees of freedom
    Multiple R-squared:  0.03482,	Adjusted R-squared:  0.03313 
    F-statistic:  20.6 on 1 and 571 DF,  p-value: 6.906e-06



### Model 2: Treatment and structure

Now let's take into account that the data have structure in the clustering of observations within manufacturer. I include random intercepts by manufacturer to account for this. Wood (2017) notes that a Bayesian interpretation of the smoothing penalty in a GAM gives the penalty the same structure as the random effects term in a mixed model. `mgcv::` exploits this to allow us to include random effects in our model using the smooth term `s()` with a basis `'re'` (i.e. `r`andom `e`ffects). In the model below, this is `s(id, bs = 're')`


```R
modTreatNoBaseline <- gam(log_profit_rug_business ~ treatment + 
    s(id, bs = "re"), data = main[dubs_sample,])

summary(modTreatNoBaseline)
```


    
    Family: gaussian 
    Link function: identity 
    
    Formula:
    log_profit_rug_business ~ treatment + s(id, bs = "re")
    
    Parametric coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  6.60723    0.06392 103.369  < 2e-16 ***
    treatment    0.22369    0.04881   4.583 5.63e-06 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Approximate significance of smooth terms:
             edf Ref.df     F p-value
    s(id) 0.2965      1 0.421   0.234
    
    R-sq.(adj) =  0.0338   Deviance explained =  3.6%
    GCV = 0.30271  Scale est. = 0.30149   n = 573


Our treatment effect estimate hasn't really changed. Nor has the amount of variance explained ('Deviance explained' is R<sup>2</sup> if a response is Gaussian). Clearly there is a lot of variance in sales profit that is explained by variables other than our treatment and the structure of our observations.

### Model 3: Treatment, structure and a covariate
It's reasonable to think that the manufacturers differed in their profits prior to being randomized to treatment, so we can add log baseline profit as a covariate to better account for any variance in profit. This is the profit for the month preceeding the first round of data collection, which is why this round of data is excluded from our data above.



```R
modTreatBaseline <- gam(log_profit_rug_business ~ treatment  + 
    log_profit_rug_business_b + s(id, bs = "re"), data = main[dubs_sample,])
summary(modTreatBaseline)
```


    
    Family: gaussian 
    Link function: identity 
    
    Formula:
    log_profit_rug_business ~ treatment + log_profit_rug_business_b + 
        s(id, bs = "re")
    
    Parametric coefficients:
                              Estimate Std. Error t value Pr(>|t|)    
    (Intercept)                6.65337    0.07583  87.746  < 2e-16 ***
    treatment                  0.26418    0.04613   5.727 1.66e-08 ***
    log_profit_rug_business_b  0.28615    0.03033   9.435  < 2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Approximate significance of smooth terms:
            edf Ref.df     F p-value
    s(id) 0.501      1 1.004   0.157
    
    R-sq.(adj) =  0.163   Deviance explained = 16.7%
    GCV = 0.26268  Scale est. = 0.26108   n = 573


So baseline profit increases the variance explained by about 13%. Notice that our treatment effect increases somewhat and its sampling variance reduces. This is consistent with the claim that baseline profit is an important covariate for our outcome. 

However, this assumes a linear relationship between profit and baseline profit. It's better to model this with a nonlinear relationship, as I do now.

### Model 4: Treatment, structure and a smoothed covariate

The relationship between log baseline profit and log monthly profit need not be linear. Here's where GAMs come in handy. It's easy to set up a smoothed basis for log baseline profit in `mgcv::`. Including the term `s(log_profit_rug_business_b)` tells the gam function to model the effect of log baseline profit with a [penalized thin plate spline](https://en.wikipedia.org/wiki/Thin_plate_spline). This tells `mgcv::` to fit a smooth function to log baseline profit that minimises the difference between the outcome $y$ and the function $f$ with a penalty $\lambda$ on the "wiggliness" of the function.
$$\mid\mid y - f \mid\mid^2 +\,\lambda\int \left(\frac{\partial^2 f(\log[\textrm{baseline profit}])}{\partial \log [\textrm{baseline profit}]^2}\right)^2dx$$


```R
modSmoothBase <- gam(
    log_profit_rug_business ~ treatment + 
    s(id, bs = "re") + s(log_profit_rug_business_b), 
    data = main[dubs_sample,]
)

summary(modSmoothBase)
```


    
    Family: gaussian 
    Link function: identity 
    
    Formula:
    log_profit_rug_business ~ treatment + s(id, bs = "re") + s(log_profit_rug_business_b)
    
    Parametric coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  6.70954    0.07997  83.901  < 2e-16 ***
    treatment    0.26134    0.04555   5.737 1.57e-08 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Approximate significance of smooth terms:
                                   edf Ref.df      F p-value    
    s(id)                        0.542  1.000  1.251   0.129    
    s(log_profit_rug_business_b) 8.580  8.947 17.230  <2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    R-sq.(adj) =  0.235   Deviance explained = 24.8%
    GCV = 0.24346  Scale est. = 0.23873   n = 573


Adding this smooth term increased the variance explained by about 8%. The models we've built so far have explained very little variance, so this is a decent increase given that all we needed to do to achieve it was reparametrize the baseline covariate. The approximate significance of this smoothed term is given in the model summary.

### Model 5: Treatment, structure and a conditional average treatment effect

Finally, we can allow the effect of treatment to vary with log baseline profit - a conditional average treatment effect. We achieve this through the use of the `by = ` argument in the smooth term.


```R
modSmoothBaseConditional <- gam(
    log_profit_rug_business ~ treatment + 
    s(id, bs = "re") + 
    s(log_profit_rug_business_b) + 
    s(log_profit_rug_business_b, by = treatment), 
    data = main[dubs_sample,]
)

summary(modSmoothBaseConditional)
```


    
    Family: gaussian 
    Link function: identity 
    
    Formula:
    log_profit_rug_business ~ treatment + s(id, bs = "re") + s(log_profit_rug_business_b) + 
        s(log_profit_rug_business_b, by = treatment)
    
    Parametric coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  6.66017    0.06557 101.572  < 2e-16 ***
    treatment    0.13150    0.02228   5.901 6.25e-09 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Approximate significance of smooth terms:
                                              edf Ref.df      F  p-value    
    s(id)                                  0.3499   1.00  0.564    0.202    
    s(log_profit_rug_business_b)           8.5178   8.93 18.656  < 2e-16 ***
    s(log_profit_rug_business_b):treatment 1.5000   1.50 34.645 8.08e-05 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Rank: 21/22
    R-sq.(adj) =  0.252   Deviance explained = 26.7%
    GCV = 0.23821  Scale est. = 0.23327   n = 573


Our variance explained increases by about 2%, not much. Our treatment effect estimate is now the effect of treatment at the mean log baseline profit.


```R
AIC(modTreatNoCovar, modTreatNoBaseline, modTreatBaseline, modSmoothBase,  modSmoothBaseConditional)
```


<table class="dataframe">
<caption>A data.frame: 5 × 2</caption>
<thead>
	<tr><th></th><th scope=col>df</th><th scope=col>AIC</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>modTreatNoCovar</th><td> 3.000000</td><td>943.4893</td></tr>
	<tr><th scope=row>modTreatNoBaseline</th><td> 3.296509</td><td>943.3619</td></tr>
	<tr><th scope=row>modTreatBaseline</th><td> 4.500987</td><td>862.0882</td></tr>
	<tr><th scope=row>modSmoothBase</th><td>12.121909</td><td>818.3478</td></tr>
	<tr><th scope=row>modSmoothBaseConditional</th><td>12.867699</td><td>805.8160</td></tr>
</tbody>
</table>



Clearly this is the best fitting model when compared to the others that I've run. However, is it better than a linear mixed model without any smoothing? 


```R
AIC(
    gam(
        log_profit_rug_business ~ log_profit_rug_business_b*treatment + s(id, bs = 're'), 
        data = main[dubs_sample,], 
    )
)
```


849.68874726521


The AIC for the smoothed conditional treatment model (806) is smaller than that for the mixed model (850), so we prefer the smoothed model. Now I can run some diagnostics on the model to check the fit.


```R
gam.check(modSmoothBaseConditional)
```

    
    Method: GCV   Optimizer: magic
    Smoothing parameter selection converged after 19 iterations.
    The RMS GCV score gradient at convergence was 5.938335e-08 .
    The Hessian was positive definite.
    Model rank =  21 / 22 
    
    Basis dimension (k) checking results. Low p-value (k-index<1) may
    indicate that k is too low, especially if edf is close to k'.
    
                                              k'   edf k-index p-value    
    s(id)                                   1.00  0.35    0.82  <2e-16 ***
    s(log_profit_rug_business_b)            9.00  8.52    1.01    0.69    
    s(log_profit_rug_business_b):treatment 10.00  1.50    1.01    0.62    
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1



    
![png](/Figures/2021-06-29/output_38_1.png)
    


The residuals and QQ plot look fine. The response vs. fitted values plot demonstrates what we already knew from the explained deviance, the model doesn't explain a lot of the variance in log profit. 

One nice property of `mgcv::`'s GAMs is that it's easy to plot the contribution of the smoothed terms to the model. The next two plots show the contribution of baseline profit and the treatment effect conditioned on baseline profit.


```R
plot(modSmoothBaseConditional, select = 2, shade = TRUE)
abline(0,0, lty = 2)
```


    
![png](/Figures/2021-06-29/output_41_0.png)
    


Each small vertical line on the x represents an observation of log baseline profit. The x-axis is centered on the mean of log baseline profit. We can see that baseline profit has a substantially nonlinear effect on profit in this sample, and in some cases is associated with reductions in profit. 

The conditional treatment effect shows a plot that is almost linear. This has a simple interpretation. The effect of treatment is monotonically decreasing with increasing baseline profit. I wouldn't put much faith in the fact that the function deviates below zero (dashed line) at high baseline profits, because there are only 3 observations there. 


```R
plot(modSmoothBaseConditional, select = 3, residuals = TRUE, shade = TRUE,)
abline(0, 0, lty =2)
```


    
![png](/Figures/2021-06-29/output_43_0.png)
    


## Conclusion

GAMs are a powerful tool for estimating any nonlinear contributions of covariates when modelling data from an A/B test (or indeed any design). I show here how to use them in the context of an A/B test to investigate the treatment effect, conditional on some covariate. This accounts for the nonlinear relationship between the covariate and the independent variable and allows us to investigate how the treatment effect varies with this covariate.

In this A/B test, we see that estimating a nonlinear relationship for baseline profit on future profits and conditioning the treatment effect on baseline profit gives us the best model. Importantly, it's better than a model with linear effects. The conditional effect reveals a decreasing relationship of baseline profit on the effect of being exposed to future markets, suggesting that this treatment may not be useful for manufacturers with above average baseline profits.
