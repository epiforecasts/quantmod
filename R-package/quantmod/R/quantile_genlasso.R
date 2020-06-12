#' Quantile generalized lasso
#'
#' Compute quantile generalized lasso solutions.
#'
#' @param x Matrix of predictors. If sparse, then passing it an appropriate 
#'   sparse \code{Matrix} class can greatly help optimization. 
#' @param y Vector of responses.
#' @param d Matrix defining the generalized lasso penalty; see details. If
#'   sparse, then passing it an appropriate sparse \code{Matrix} class can 
#'   greatly help optimization. A convenience function \code{get_diff_mat} for 
#'   constructing trend filtering penalties is provided. 
#' @param tau,lambda Vectors of quantile levels and tuning parameter values. If
#'   these are not of the same length, the shorter of the two is recycled so
#'   that they become the same length. Then, for each \code{i}, we solve a
#'   separate quantile generalized lasso problem at quantile level \code{tau[i]}
#'   and tuning parameter value \code{lambda[i]}. The most common use cases are:
#'   specifying one tau value and a sequence of lambda values; or specifying a
#'   sequence of tau values and one lambda value.
#' @param weights Vector of observation weights (to be used in the loss
#'   function). Default is NULL, which is interpreted as a weight of 1 for each
#'   observation.  
#' @param no_pen_rows Indices of the rows of \code{d} that should be excluded 
#'   from the generalized lasso penalty. Default is \code{c()}, which means that
#'   no rows are to be excluded.
#' @param intercept Should an intercept be included in the regression model?
#'   Default is TRUE.
#' @param standardize Should the predictors be standardized (to have zero mean
#'   and unit variance) before fitting?  Default is TRUE.
#' @param noncross Should noncrossing constraints be applied? These force the
#'   estimated quantiles to be properly ordered across all quantile levels being
#'   considered. The default is FALSE. If TRUE, then noncrossing constraints are
#'   applied to the estimated quantiles at all points specified by the next
#'   argument \code{x0}. Note: this option only makes sense if the values in the
#'   \code{tau} vector are distinct, and sorted in increasing order.
#' @param x0 Matrix of points used to define the noncrossing
#'   constraints. Default is NULL, which means that we consider noncrossing
#'   constraints at the training points \code{x}.
#' @param lp_solver One of "gurobi" or "glpk", indicating which LP solver to
#'   use. Default is "gurobi".
#' @param time_limit This sets the maximum amount of time (in seconds) to allow
#'   Gurobi or GLPK to solve any single quantile generalized lasso problem (for
#'   a single tau and lambda value). Default is NULL, which means unlimited
#'   time.
#' @param warm_starts Should warm starts be used in the LP solver (from one LP 
#'   solve to the next)? Only supported for Gurobi.
#' @param params A list of control parameters to pass to Gurobi or GLPK. Default
#'   is \code{list()} which means no additional parameters are passed. For
#'   example: with Gurobi, we can use \code{list(Threads=4)} to specify that  
#'   Gurobi should use 4 threads when available. (Note that if a time limit is
#'   specified through this \code{params} list, then its value will be overriden 
#'   by the last argument \code{time_limit}, assuming the latter is not NULL.) 
#' @param transform,inv_trans The first is a function to transform y before
#'   solving the quantile generalized lasso; the second is the corresponding
#'   inverse transform. For example: for count data, we might want to model
#'   log(1+y) (which would be the transform, and the inverse transform would be
#'   exp(x)-1). Both \code{transform} and \code{inv_trans} should be
#'   vectorized. Convenience functions \code{log_pad} and \code{inv_log_pad} are
#'   provided.
#' @param jitter Function for applying random jitter to y, which might help
#'   optimization. For example: for count data, there can be lots of ties (with
#'   or without transformation of y), which can make optimization more
#'   difficult. The function \code{jitter} should take an integer n and return n
#'   random draws. A convenience function \code{unif_jitter} is provided.
#' @param verbose Should progress be printed out to the console? Default is
#'   FALSE.
#'
#' @return A list with the following components:
#'   \itemize{
#'   \item beta: a matrix of generalized lasso coefficients, of dimension =
#'   (number of features + 1) x (number of quantile levels) assuming
#'   \code{intercept=TRUE}, else (number of features) x (number of quantile
#'   levels). Note: these coefficients will always be on the appropriate scale;
#'   they are always on the scale of original features, even if
#'   \code{standardize=TRUE}
#'   \item status: vector of status flags returned by Gurobi's or GLPK's LP
#'   solver, of length = (number of quantile levels)
#'   \item tau, lambda: vectors of tau and lambda values used
#'   \item weights, no_pen_rows, ..., jitter: values of these other arguments
#'   used  in the function call  
#'   }
#'
#' @details This function solves the quantile generalized lasso problem, for
#'   each pair of quantile level \eqn{\tau} and tuning parameter \eqn{\lambda}: 
#'   \deqn{\mathop{\mathrm{minimize}}_{\beta_0,\beta} \;
#'   \sum_{i=1}^n w_i \psi_\tau(y_i-\beta_0-x_i^T\beta) + \lambda \|D\beta\|_1}   
#'   for a response vector \eqn{y} with components \eqn{y_i}, predictor matrix
#'   \eqn{X} with rows \eqn{x_i}, and penalty matrix \eqn{D}. Here
#'   \eqn{\psi_\tau(v) = \max\{\tau v, (\tau-1) v\}} is the  
#'   "pinball" or "tilted \eqn{\ell_1}" loss. When noncrossing constraints are
#'   applied, we instead solve one big joint optimization, over all quantile
#'   levels and tuning parameter values: 
#'   \deqn{\mathop{\mathrm{minimize}}_{\beta_{0k}, \beta_k, k=1,\ldots,r} \;
#'   \sum_{k=1}^r \bigg(\sum_{i=1}^n w_i \psi_{\tau_k}(y_i-\beta_{0k}-
#'   x_i^T\beta_k) + \lambda_k \|D\beta_k\|_1\bigg)} 
#'   \deqn{\mathrm{subject \; to} \;\; \beta_{0k}+x^T\beta_k \leq
#'   \beta_{0,k+1}+x^T\beta_{k+1} \;\; k=1,\ldots,r-1, \; x \in \mathcal{X}}
#'   where the quantile levels \eqn{\tau_k, k=1,\ldots,r} are assumed to be in
#'   increasing order, and \eqn{\mathcal{X}} is a collection of points over
#'   which to enforce the noncrossing constraints.
#'
#'   Either problem is readily converted into a linear program (LP), and solved
#'   using either Gurobi (which is free for academic use, and generally fast) or 
#'   GLPK (which free for everyone, but slower).
#'
#' @author Ryan Tibshirani
#' @export

quantile_genlasso = function(x, y, d, tau, lambda, weights=NULL,
                             no_pen_rows=c(), intercept=TRUE, standardize=TRUE,
                             noncross=FALSE, x0=NULL,
                             lp_solver=c("gurobi", "glpk"), time_limit=NULL,  
                             warm_starts=TRUE, params=list(), transform=NULL,
                             inv_trans=NULL, jitter=NULL, verbose=FALSE) {
  # Set up some basics
  x = as.matrix(x)
  y = as.numeric(y)
  d = as.matrix(d)
  n = nrow(x)
  p = ncol(x)
  m = nrow(d)
  if (is.null(weights)) weights = rep(1,n)
  lp_solver = match.arg(lp_solver)
  if (noncross) warning("Noncrossing constraints currently not implemented!")

  # Standardize the columns of x, if we're asked to
  if (standardize) {
    bx = apply(x,2,mean)
    sx = apply(x,2,sd)
    x = scale(x,bx,sx)
  }

  # Add all 1s column to x, and all 0s column to d, if we need to 
  if (intercept) {
    x = cbind(rep(1,n), x)
    d = cbind(rep(0,m), d)
    p = p+1
  }
    
  # Transform y, if we're asked to
  if (!is.null(transform)) y = transform(y)
  
  # Recycle tau or lambda so that they're the same length
  if (length(tau) != length(lambda)) {
    k = max(length(tau), length(lambda))
    tau = rep(tau, length=k)
    lambda = rep(lambda, length=k)
  }
  
  # Solve the quantile generalized lasso LPs 
  obj = quantile_genlasso_lp(x=x, y=y, d=d, tau=tau, lambda=lambda,
                             weights=weights, no_pen_rows=no_pen_rows,
                             lp_solver=lp_solver, time_limit=time_limit,
                             warm_starts=warm_starts, params=params,
                             jitter=jitter, verbose=verbose)
 
  # Transform beta back to original scale, if we standardized
  if (standardize) {
    if (!intercept) obj$beta = rbind(rep(0,length(tau)), obj$beta)
    obj$beta[-1,] = Diagonal(x=1/sx) %*% obj$beta[-1,] 
    obj$beta[1,] = obj$beta[1,] - (bx/sx) %*% obj$beta[-1,] 
  }
  
  colnames(obj$beta) = sprintf("t=%g, l=%g", tau, lambda)
  obj = c(obj, enlist(tau, lambda, weights, no_pen_rows, intercept, standardize,  
                      lp_solver, warm_starts, time_limit, params, transform,
                      inv_trans, jitter))
  class(obj) = "quantile_genlasso"
  return(obj)
}

# Solve quantile generalized lasso problems using an LP solver.

quantile_genlasso_lp = function(x, y, d, tau, lambda, weights, no_pen_rows,
                                lp_solver="gurobi", params=list(),
                                warm_starts=TRUE, time_limit=time_limit,  
                                jitter=NULL, verbose=FALSE){  
  # Set up some basic objects that we will need
  n = nrow(x); p = ncol(x); m = nrow(d)
  Inn = Diagonal(n); Imm = Diagonal(m)
  Znm = Matrix(0,n,m,sparse=TRUE)
  Zmn = Matrix(0,m,n,sparse=TRUE)
  model = list()
  model$sense = rep(">=", 2*n+2*m)

  # Gurobi setup
  if (lp_solver == "gurobi") {
    if (!require("gurobi",quietly=TRUE)) { 
      stop("Package gurobi not installed (required here)!")
    }
    if (!is.null(time_limit)) params$TimeLimit = time_limit
    if (is.null(params$LogToConsole)) params$LogToConsole = 0
    if (verbose && length(tau) == 1) params$LogToConsole = 1
    model$lb = c(rep(-Inf,p), rep(0,m), rep(0,n))
  }

  # GLPK setup
  else if (lp_solver == "glpk") {
    if (!require("Rglpk",quietly=TRUE)) { 
      stop("Package Rglpk not installed (required here)!")
    }
    if (!is.null(time_limit)) params$tm_limit = time_limit * 1000
    if (verbose && length(tau) == 1) params$verbose = TRUE
    model$bounds = list(lower=list(ind=1:p, val=rep(-Inf,p)))
  }
  
  # Loop over tau/lambda values
  beta = Matrix(0, nrow=p, ncol=length(tau), sparse=TRUE)
  status = rep(NA, length(tau))
  last_sol = NULL
  
  if (verbose && length(tau) > 1) {
    cat(sprintf("Problems solved (of %i): ", length(tau)))
  }
  for (j in 1:length(tau)) {
    if (verbose && length(tau) > 1 && (length(tau) <= 10 || j %% 5 == 0)) {
      cat(paste(j, "... "))
    }

    # Apply random jitter, if we're asked to
    if (!is.null(jitter)) yy = y + jitter(n)
    else yy = y

    # Vector of objective coefficients
    model$obj = c(rep(0,p), rep(lambda[j],m), weights)
    model$obj[p + no_pen_rows] = 0 # No L1 penalty on excluded rows
    
    # Matrix of constraint coefficients: depends only on tau, so we try to save
    # work if possible (check if we've already created this for last tau value) 
    if (j == 1 || tau[j] != tau[j-1]) {
      model$A = rbind(
        cbind(tau[j]*x, Znm, Inn),
        cbind((tau[j]-1)*x, Znm, Inn),
        cbind(-d, Imm, Zmn),
        cbind(d, Imm, Zmn)
      )
    }

    # Right hand side of constraints
    model$rhs = c(tau[j]*y, (tau[j]-1)*y, rep(0,2*m))

    # Gurobi
    if (lp_solver == "gurobi") {
      # Set a warm start, if we're asked to
      if (warm_starts && !is.null(last_sol)) {
        model$start = last_sol
      }
    
      # Call Gurobi's LP solver, store results
      a = gurobi(model=model, params=params)
      beta[,j] = a$x[1:p] 
      status[j] = a$status
      if (warm_starts) last_sol = a$x
    }

    # GLPK
    else if (lp_solver == "glpk") {
      # Call GLPK's LP solver, store results
      a = Rglpk_solve_LP(obj=model$obj, mat=model$A, dir=model$sense,
                         rhs=model$rhs, bounds=model$bounds, control=params)  
      beta[,j] = a$solution[1:p] 
      status[j] = a$status
    }
  }; if (verbose && length(tau) > 1) cat("\n")
  
  return(enlist(beta, status))
}

##############################

#' Coef function for quantile_genlasso object
#'
#' Retrieve coefficients for estimating the conditional quantiles, using the
#' generalized lasso coefficients at particular tau or lambda values. 
#' 
#' @param obj The \code{quantile_genlasso} object.
#' @param s Vector of integers specifying the tau and lambda values to consider
#'   for predictions; for each \code{i} in this vector, a prediction is made at 
#'   quantile level \code{tau[i]} and tuning parameter value \code{lambda[i]},
#'   according to the \code{tau} and \code{lambda} vectors stored in the given  
#'   \code{quantile_genlasso} object \code{obj}. (Said differently, \code{s} 
#'   specifies the columns of \code{obj$beta} to use for the predictions.)
#'   Default is NULL, which means that all tau and lambda values will be
#'   considered.
#' 
#' @export

coef.quantile_genlasso = function(obj, s=NULL) {
  if (is.null(s)) s = 1:ncol(obj$beta)
  return(obj$beta[,s])
}

##############################

#' Predict function for quantile_genlasso object
#'
#' Predict the conditional quantiles at a new set of predictor variables, using
#' the generalized lasso coefficients at particular tau or lambda values.
#' 
#' @param obj The \code{quantile_genlasso} object.
#' @param newx Matrix of new predictor variables at which predictions should
#'   be made; if missing, the original (training) predictors are used.
#' @param s Vector of integers specifying the tau and lambda values to consider
#'   for predictions; for each \code{i} in this vector, a prediction is made at 
#'   quantile level \code{tau[i]} and tuning parameter value \code{lambda[i]},
#'   according to the \code{tau} and \code{lambda} vectors stored in the given  
#'   \code{quantile_genlasso} object \code{obj}. (Said differently, \code{s} 
#'   specifies the columns of \code{object$beta} to use for the predictions.)
#'   Default is NULL, which means that all tau and lambda values will be
#'   considered. 
#' @param sort Should the quantile estimates be sorted? Default is FALSE. Note:
#'   this option only makes sense if the values in the stored \code{tau} vector
#'   are distinct, and sorted in increasing order.  
#' @param iso Should the quantile estimates be passed through isotonic
#'   regression? Default is FALSE; if TRUE, takes priority over
#'   \code{sort}. Note: this option only makes sense if the values in the stored 
#'   \code{tau} vector are distinct, and sorted in increasing order.  
#' @param nonneg: should the quantile estimates be truncated at 0? Natural for
#'   count data. Default is FALSE. 
#' @param round: should the quantile estimates be rounded? Natural for count
#'   data. Default is FALSE.
#' 
#' @export

predict.quantile_genlasso = function(obj, newx, s=NULL, sort=FALSE, iso=FALSE,
                                     nonneg=FALSE, round=FALSE) {
  newx = as.matrix(newx); n0 = nrow(newx)
  if (obj$intercept || obj$standardize) newx = cbind(rep(1,n0), newx) 
  z = as.matrix(newx %*% coef(obj,s))

  # Apply the inverse transform, if we're asked to
  if (!is.null(obj$inv_trans)) {
    # Annoying, must handle carefully the case that z drops to a vector 
    names = colnames(z)
    z = apply(z, 2, obj$inv_trans)
    z = matrix(z, nrow=n0)
    colnames(z) = names
  }
  
  # Run isotonic regression, sort, truncated, round, if we're asked to
  for (i in 1:nrow(z)) {
    if (sort) z[i,] = sort(z[i,])
    if (iso) z[i,] = isoreg(z[i,])$yf
  }
  if (nonneg) z = pmax(z,0)
  if (round) z = round(z)
  return(z)
}

##############################

#' Quantile generalized lasso on a tau by lambda grid
#'
#' Convenience function for computing quantile generalized lasso solutions on a
#' tau by lambda grid. 
#'
#' @param nlambda Number of lambda values to consider, for each quantile
#'   level. Default is 30.  
#' @param lambda_min_ratio Ratio of the minimum to maximum lambda value, for
#'   each quantile levels. Default is 1e-3.
#'
#' @details This function forms a \code{lambda} vector either determined by the
#'   \code{nlambda} and \code{lambda_min_ratio} arguments, or the \code{lambda}
#'   argument; if the latter is specified, then it takes priority. Then, for
#'   each \code{i} and \code{j}, we solve a separate quantile generalized lasso
#'   problem at quantile level \code{tau[i]} and tuning parameter value
#'   \code{lambda[j]}, using the \code{quantile_genlasso} function. All
#'   arguments (aside from \code{nlambda} and \code{lambda_min_ratio}) are as in
#'   the latter function; noncrossing constraints are disallowed.
#' 
#' @export

quantile_genlasso_grid = function(x, y, d, tau, lambda=NULL, nlambda=30,
                                  lambda_min_ratio=1e-3, weights=NULL,
                                  no_pen_rows=c(), intercept=TRUE,
                                  standardize=TRUE,
                                  lp_solver=c("gurobi","glpk"), time_limit=NULL, 
                                  warm_starts=TRUE, params=list(),
                                  transform=NULL, inv_trans=NULL, jitter=NULL,
                                  verbose=FALSE) {
  # Set the lambda sequence, if we need to
  if (is.null(lambda)) lambda = get_lambda_seq(x, y, d, nlambda,
                                               lambda_min_ratio) 

  # Create the grid: stack the problems so that tau is constant and lambda is
  # changing from one to the next, the way we've setup the LP solver, this will 
  # be better for memory purposes (and also warm starts?)
  tau = rep(tau, each=length(lambda))
  lambda = rep(lambda, length(unique(tau)))

  # Now just call quantile_genlasso 
  obj = quantile_genlasso(x=x, y=y, d=d, tau=tau, lambda=lambda,
                          weights=weights, no_pen_rows=no_pen_rows,
                          intercept=intercept, standardize=standardize,
                          noncross=FALSE, x0=NULL, lp_solver=lp_solver,
                          time_limit=time_limit, warm_starts=warm_starts,
                          params=params, transform=transform,
                          inv_trans=inv_trans, jitter=jitter, verbose=verbose)
  class(obj) = c("quantile_genlasso_grid", class(obj))
  return(obj)
}

##############################

#' Lambda max for quantile generalized lasso 
#'
#' Compute lambda max for a quantile generalized lasso problem. 
#'
#' @details This is a rough heuristic derived from fiddling with the KKT
#'   conditions when tau = 1/2. It should be possible to improve this. If
#'   \code{d} is not specified, we will set it equal to the identity (hence
#'   interpret the problem as a quantile lasso problem).
#'
#' @export

get_lambda_max = function(x, y, d=NULL) {
  # Define a diagonal penalty matrix, if we need to
  if (is.null(d)) { x = as.matrix(x); d = Diagonal(ncol(x)) }
  
  # NB: I have **no idea** why I need to use Matrix::colSums() here. Somehow it
  # can break without it (very weird binding issue?)   
  return((1/2) * max(abs(t(x) %*% sign(y))) / median(Matrix::colSums(abs(d))))
}

#' Lambda sequence for quantile generalized lasso 
#'
#' Compute a lambda sequence for a quantile generalized lasso problem.
#'
#' @details This function returns \code{nlambda} values log-spaced in between
#'   \code{lambda_max}, as computed by \code{get_lambda_max}, and
#'   \code{lamdba_max * lambda_min_ratio}. If \code{d} is not specified, we will
#'   set it equal to the identity (hence interpret the problem as a quantile
#'   lasso problem).
#'
#' @export

get_lambda_seq = function(x, y, d=NULL, nlambda, lambda_min_ratio) { 
  # Define a diagonal penalty matrix, if we need to
  if (is.null(d)) { x = as.matrix(x); d = Diagonal(ncol(x)) }
  
  lambda_max = get_lambda_max(x, y, d)
  return(exp(seq(log(lambda_max), log(lambda_max * lambda_min_ratio),
                 length=nlambda)))
}

##############################

#' Predict function for quantile_genlasso_grid object
#' 
#' Predict the conditional quantiles at a new set of predictor variables, using
#' the generalized lasso coefficients at particular tau or lambda values.
#'
#' @details This function operates as in the \code{predict.quantile_genlasso} 
#'   function for a \code{quantile_genlasso} object, but with a few key
#'   differences. First, the output is reformatted so that it is an array of
#'   dimension (number of prediction points) x (number of tuning parameter
#'   values) x (number of quantile levels). This output is generated from the
#'   full set of tau and lambda pairs stored in the given
#'   \code{quantile_genlasso_grid} object \code{obj} (selecting a subset is 
#'   disallowed). Second, the arguments \code{sort} and \code{iso} operate on
#'   the appropriate slices of this array: for a fixed lambda value, we sort or
#'   run isotonic regression across all tau values.
#' 
#' @export

predict.quantile_genlasso_grid = function(obj, newx, sort=FALSE, iso=FALSE, 
                                          nonneg=FALSE, round=FALSE) { 
  newx = as.matrix(newx); n0 = nrow(newx)
  if (obj$intercept || obj$standardize) newx = cbind(rep(1,n0), newx)
  z = as.matrix(newx %*% coef(obj))

  # Apply the inverse transform, if we're asked to
  if (!is.null(obj$inv_trans)) {
    # Annoying, must handle carefully the case that z drops to a vector 
    names = colnames(z)
    z = apply(z, 2, obj$inv_trans)
    z = matrix(z, nrow=n0)
    colnames(z) = names
  }
  
  # Now format into an array
  z = array(z, dim=c(n0, length(unique(obj$lambda)), length(unique(obj$tau))))
  dimnames(z)[[2]] = sprintf("l=%g", unique(obj$lambda))
  dimnames(z)[[3]] = sprintf("t=%g", unique(obj$tau))

  # Run isotonic regression, sort, truncated, round, if we're asked to
  for (i in 1:dim(z)[1]) {
    for (j in 1:dim(z)[2]) {
      if (sort) z[i,j,] = sort(z[i,j,])
      if (iso) z[i,j,] = isoreg(z[i,j,])$yf
    }
  }
  if (nonneg) z = pmax(z,0)
  if (round) z = round(z)
  return(z)
}

##############################

#' Quantile generalized lasso objective
#'
#' Compute generalized lasso objective for a single tau and lambda value.    
#'
#' @export

quantile_genlasso_objective = function(x, y, d, beta, tau, lambda) {  
  loss = quantile_loss(x %*% beta, y, tau)
  pen = lambda * sum(abs(d %*% beta))
  return(loss + pen)
}
