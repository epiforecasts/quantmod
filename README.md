# quantmod

Tools for quantile modeling: penalized quantile regression, time series,
cross-validation, and ensembles. 

### Install the R package

To install the quantmod R package directly from github, run the following in R:

```{r}
library(devtools)
install_github(repo="cmu-delphi/quantmod", subdir="R-package/quantmod")
```

### Install Gurobi


... intstructions for R <= 3.6

First make sure you have Gurobi 9.0.2 installed

https://upload.gurobi.com/gurobiR/gurobi9.0.2_R.tar.gz

make -f Makefile.mac

R CMD INSTALL gurobi_9.0.2.tgz

