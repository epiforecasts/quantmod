#' Quantile ensemble
#'
#' Fit ensemble weights, given a set of quantile predictions.
#'
#' @param qarr Array of predicted quantiles, of dimension (number of prediction 
#'   points) x (number or ensemble components) x (number of quantile levels).
#' @param y Vector of responses (whose quantiles are being predicted by
#'   \code{qarr}).  
#' @param tau Vector of quantile levels at which predictions are made. 
#' @param weights Vector of observation weights (to be used in the loss
#'   function). Default is NULL, which is interpreted as a weight of 1 for each
#'   observation.
#' @param tau_groups Vector of group labels, having the same length as
#'   \code{tau}. Common labels indicate that the ensemble weights for the
#'   corresponding quantile levels should be tied together. Default is 
#'   \code{rep(1,length(tau))}, which means that a common set of ensemble
#'   weights should be used across all levels. See details. 
#' @param intercept Should an intercept be included in the ensemble model?
#'   Default is FALSE.
#' @param nonneg Should the ensemble weights be constrained to be nonnegative?
#'   Default is TRUE.
#' @param unit_sum Should the ensemble weights be constrained to sum to 1?
#'   Default is TRUE.
#' @param noncross Should noncrossing constraints be enforced? Default is
#'   TRUE. Note: this option only matters when there is more than group of
#'   ensemble weights, as determined by \code{tau_groups}. See details. 
#' @param q0 Array of points used to define the noncrossing
#'   constraints. Must haev dimension (number of points) x (number of ensemble
#'   components) x (number of quantile levels). Default is NULL, which means
#'   that we consider noncrossing constraints at the training points
#'   \code{qarr}. 
#' @param lp_solver One of "gurobi" or "glpk", indicating which LP solver to
#'   use. Default is "gurobi". 
#' @param time_limit This sets the maximum amount of time (in seconds) to allow
#'   Gurobi or GLPK to solve any single quantile generalized lasso problem (for
#'   a single tau and lambda value). Default is NULL, which means unlimited
#'   time.
#' @param params A list of control parameters to pass to Gurobi or GLPK. Default
#'   is \code{list()} which means no additional parameters are passed. For
#'   example: with Gurobi, we can use \code{list(Threads=4)} to specify that  
#'   Gurobi should use 4 threads when available. (Note that if a time limit is
#'   specified through this \code{params} list, then its value will be overriden 
#'   by the last argument \code{time_limit}, assuming the latter is not NULL.) 
#' @param verbose Should progress be printed out to the console? Default is
#'   FALSE.
#' 
#' @return A list with the following components:
#'   \itemize{
#'   \item alpha: vector or matrix of ensemble weights. If \code{tau_groups} has
#'   only one unique label, then this is a vector of length = (number of
#'   ensemble components); otherwise, it is a matrix, of dimension (number of
#'   ensemble components) x (number of quantile levels)
#'   \item tau: vector of quantile levels used
#'   \item weights, tau_groups, ..., params: values of these other arguments
#'   used in the function call  
#'   }
#' 
#' @details This function solves the following quantile ensemble optimization
#'   problem, over quantile levels \eqn{\tau_k, k=1,\ldots,r}:
#'   \deqn{\mathop{\mathrm{minimize}}_{\alpha_j, j=1,\ldots,p} \; \sum_{k=1}^r 
#'   \sum_{i=1}^n w_i \psi_{\tau_k} \bigg(y_i - \sum_{j=1}^p \alpha_j q_{ijk}
#'   \bigg)} \deqn{\mathrm{subject \; to} \;\; \sum_{j=1}^p \alpha_j = 1, \; 
#'   \alpha_j \geq 0, \; j=1,\ldots,p}
#'   for a response vector \eqn{y} and quantile array \eqn{q}, where
#'   \eqn{q_{ijk}} is an estimate of the quantile of \eqn{y_i} at the level
#'   \eqn{\tau_k}, from ensemble component member \eqn{j}. Here
#'   \eqn{\psi_\tau(v) = \max\{\tau v, (\tau-1) v\}} is the "pinball" or "tilted
#'   \eqn{\ell_1}" loss. A more advanced version allows us to estimate a
#'   separate ensemble weight \eqn{\alpha_{jk}} per component method \eqn{j},
#'   per quantile level \eqn{k}: 
#'   \deqn{\mathop{\mathrm{minimize}}{\alpha_{jk}, j=1,\ldots,p, k=1,\ldots,r} 
#'   \; \sum_{k=1}^r \sum_{i=1}^n w_i \psi_{\tau_k} \bigg(y_i - \sum_{j=1}^p
#'   \alpha_{jk} q_{ijk} \bigg)} \deqn{\mathrm{subject \; to} \;\; 
#'   \sum_{j=1}^p \alpha_{jk} = 1, \; k=1,\ldots,r, \; 
#'   \alpha_{jk} \geq 0, \; j=1,\ldots,p, \; k=1,\ldots,r}
#'   As a form of regularization, we can additionally incorporate noncrossing
#'   constraints into the above optimization, which take the form:
#'   \deqn{\alpha_{\bullet,k}^T q \leq \alpha_{\bullet,k+1}^T q, \;
#'   k=1,\ldots,r-1, \; q \in \mathcal{Q}}
#'   where the quantile levels \eqn{\tau_k, k=1,\ldots,r} are assumed to be in
#'   increasing order, and \eqn{\mathcal{Q}} is a collection of points over
#'   which to enforce the noncrossing constraints. Finally, somewhere in between
#'   these two extremes is to allow one ensemble weight per component member
#'   \eqn{j}, per quantile group \eqn{g}. This can be interpreted as a set of
#'   further constraints which enforce equality between \eqn{\alpha_{jk}} and
#'   \eqn{\alpha_{j\ell}}, for all \eqn{k,\ell} that are in the same group
#'   \eqn{g}.  
#'
#' @export

quantile_ensemble = function(qarr, y, tau, weights=NULL,
                             tau_groups=rep(1,length(tau)), intercept=FALSE,
                             nonneg=TRUE, unit_sum=TRUE, noncross=TRUE, q0=NULL, 
                             lp_solver="gurobi", time_limit=NULL, params=list(),
                             verbose=FALSE) {  
  # Set up some basics
  qarr = as.array(qarr)
  y = as.numeric(y)
  n = dim(qarr)[1]
  p = dim(qarr)[2]
  r = dim(qarr)[3] 
  if (is.null(weights)) weights = rep(1,n)

  # Add an all 1s matrix to qarr, if we need to
  if (intercept) {
    a = array(NA, dim=c(n,p+1,r))
    for (k in 1:r) a[,,k] = cbind(rep(1,n), qarr[,,k])  
    qarr = a
    p = p+1
  }

  # Standard stacking
  if (length(unique(tau_groups)) == 1) {
    obj = quantile_ensemble_stand(qarr=qarr, y=y, tau=tau, weights=weights, 
                                  intercept=intercept, nonneg=nonneg,
                                  unit_sum=unit_sum, lp_solver=lp_solver,
                                  time_limit=time_limit, params=params, verbose)
  }
  # Flexible stacking
  else {
    obj = quantile_ensemble_flex(qarr=qarr, y=y, tau=tau, weights=weights,
                                 tau_groups=tau_groups, intercept=intercept,
                                 nonneg=nonneg, unit_sum=unit_sum,
                                 noncross=noncross, q0=q0, lp_solver=lp_solver,
                                 time_limit=time_limit, params=params, verbose)
    colnames(obj$alpha) = tau
  }

  obj = c(obj, enlist(tau, weights, tau_groups, intercept, nonneg, unit_sum,
                      noncross, q0, lp_solver, time_limit, params))   
  class(obj) = "quantile_ensemble"
  return(obj)
}

# Solve standard quantile stacking using an LP solver.
  
quantile_ensemble_stand = function(qarr, y, tau, weights, intercept=FALSE,
                                   nonneg=TRUE, unit_sum=TRUE,
                                   lp_solver="gurobi", time_limit=NULL, 
                                   params=list(), verbose=FALSE) { 
  # Set up some basic objects that we will need
  n = dim(qarr)[1]
  p = dim(qarr)[2]
  r = dim(qarr)[3]; N = n*r
  INN = Diagonal(N)
  model = list()

 # Gurobi setup
  if (lp_solver == "gurobi") {
    if (!require("gurobi",quietly=TRUE)) { 
      stop("Package gurobi not installed (required here)!")
    }
    if (!is.null(time_limit)) params$TimeLimit = time_limit
    if (is.null(params$LogToConsole)) params$LogToConsole = 0
    if (verbose) params$LogToConsole = 1
    equal_sign = "="
  }

  # GLPK setup
  else if (lp_solver == "glpk") {
    if (!require("Rglpk",quietly=TRUE)) { 
      stop("Package Rglpk not installed (required here)!")
    }
    if (!is.null(time_limit)) params$tm_limit = time_limit * 1000
    if (verbose) params$verbose = TRUE
    equal_sign = "=="
  }
  
  # Vector of objective coefficients
  model$obj = c(rep(0,p), rep(weights, each=r))
  
  # Matrix of constraint coefficients
  model$A = Matrix(0, nrow=2*N, ncol=p+N, sparse=TRUE)
  model$sense = model$rhs = rep(NA, 2*N)
  
  # First part of residual constraint
  for (k in 1:r) { 
    model$A[(k-1)*n + 1:n, 1:p] = tau[k] * qarr[,,k]
    model$sense[(k-1)*n + 1:n] = ">="
    model$rhs[(k-1)*n + 1:n] = tau[k] * y
  }
  model$A[1:N, p + 1:N] = INN
  
  # Second part of residual constraint
  for (k in 1:r) {
    model$A[(r+k-1)*n + 1:n, 1:p] = (tau[k]-1) * qarr[,,k]
    model$sense[(r+k-1)*n + 1:n] = ">="
    model$rhs[(r+k-1)*n + 1:n] = (tau[k]-1) * y
  }
  model$A[N + 1:N, p + 1:N] = INN
  
  # Unit sum constraints on alpha, if we're asked to
  if (unit_sum) {
    vec = rep(1,p); if (intercept) vec[1] = 0
    model$A = rbind(model$A, c(vec, rep(0,N)))
    model$sense = c(model$sense, equal_sign)
    model$rhs = c(model$rhs, 1)
  }

  # Remove nonnegativity constraint on alpha, if we're asked to
  if (!nonneg) {
    if (lp_solver == "gurobi") model$lb = c(rep(-Inf,p), rep(0,N))
    if (lp_solver == "glpk") model$bounds = list(lower=list(ind=1:p,
                                                            val=rep(-Inf,p)))
  }
  
  # Remove nonnegativity constraint on intercept, if needed
  if (intercept && nonneg) {
    if (lp_solver == "gurobi") model$lb = c(-Inf, rep(0,p-1+N))
    if (lp_solver == "glpk") model$bounds = list(lower=list(ind=1, val=-Inf))
  }

  # Call Gurobi's LP solver, store results
  if (lp_solver == "gurobi") {
    a = gurobi(model=model, params=params)
    alpha = a$x[1:p] 
    status = a$status
  }

  # Call GLPK's LP solver, store results
  else if (lp_solver == "glpk") {
    a = Rglpk_solve_LP(obj=model$obj, mat=model$A, dir=model$sense,
                       rhs=model$rhs, bounds=model$bounds, control=params)  
    alpha = a$solution[1:p] 
    status = a$status
  }

  return(enlist(alpha, status))
}

# Solve flexible quantile stacking using an LP solver.

quantile_ensemble_flex = function(qarr, y, tau, weights, tau_groups,
                                  intercept=FALSE, nonneg=TRUE, unit_sum=TRUE,
                                  noncross=FALSE, q0=NULL, lp_solver="gurobi", 
                                  time_limit=NULL, params=list(),
                                  verbose=FALSE) { 
  # Set up some basic objects that we will need
  n = dim(qarr)[1]
  p = dim(qarr)[2]
  r = dim(qarr)[3]
  N = n*r; P=p*r 
  INN = Diagonal(N)
  model = list()

 # Gurobi setup
  if (lp_solver == "gurobi") {
    if (!require("gurobi",quietly=TRUE)) { 
      stop("Package gurobi not installed (required here)!")
    }
    if (!is.null(time_limit)) params$TimeLimit = time_limit
    if (is.null(params$LogToConsole)) params$LogToConsole = 0
    if (verbose) params$LogToConsole = 1
    equal_sign = "="
  }

  # GLPK setup
  else if (lp_solver == "glpk") {
    if (!require("Rglpk",quietly=TRUE)) { 
      stop("Package Rglpk not installed (required here)!")
    }
    if (!is.null(time_limit)) params$tm_limit = time_limit * 1000
    if (verbose) params$verbose = TRUE
    equal_sign = "=="
  }
  
  # Vector of objective coefficients
  model$obj = c(rep(0,P), rep(weights, each=r))
  
  # Matrix of constraint coefficients
  model$A = Matrix(0, nrow=2*N, ncol=P+N, sparse=TRUE)
  model$sense = model$rhs = rep(NA, 2*N)
  
  # First part of residual constraint
  for (k in 1:r) { 
    model$A[(k-1)*n + 1:n, (k-1)*p + 1:p] = tau[k] * qarr[,,k]
    model$sense[(k-1)*n + 1:n] = ">="
    model$rhs[(k-1)*n + 1:n] = tau[k] * y
  }
  model$A[1:N, P + 1:N] = INN
  
  # Second part of residual constraint
  for (k in 1:r) {
    model$A[(r+k-1)*n + 1:n, (k-1)*p + 1:p] = (tau[k]-1) * qarr[,,k]
    model$sense[(r+k-1)*n + 1:n] = ">="
    model$rhs[(r+k-1)*n + 1:n] = (tau[k]-1) * y
  }
  model$A[N + 1:N, P + 1:N] = INN
  
  # Group equality constraints, if needed
  labels = unique(tau_groups); num_labels = length(labels)
  if (num_labels < r) {
    Ipp = Diagonal(p)
    nrowA = nrow(model$A)
    model$A = rbind(model$A, Matrix(0, nrow=p*(r-num_labels), ncol=P+N, sparse=TRUE))
    for (label in labels) {
      ind = which(tau_groups == label)
      if (length(ind) > 1) { 
        for (k in 1:(length(ind)-1)) {
          model$A[nrowA + (k-1)*p + 1:p, (ind[k]-1)*p + 1:p] = -Ipp
          model$A[nrowA + (k-1)*p + 1:p, (ind[k+1]-1)*p + 1:p] = Ipp
          model$sense[nrowA + (k-1)*p + 1:p] = equal_sign
          model$rhs[nrowA + (k-1)*p + 1:p] = 0
        }
        nrowA = nrowA + (length(ind)-1)*p
      }
    }
  }
  
  # Unit sum constraints on alpha, if we're asked to
  if (unit_sum) {
    vec = rep(1,p); if (intercept) vec[1] = 0
    nrowA = nrow(model$A)
    model$A = rbind(model$A, Matrix(0, nrow=r, ncol=P+N, sparse=TRUE))
    for (k in 1:r) model$A[nrowA + k, (k-1)*p + 1:p] = vec
    model$sense = c(model$sense, rep(equal_sign, r))
    model$rhs = c(model$rhs, rep(1, r))
  }
  
  # Remove nonnegativity constraint on alpha, if we're asked to
  if (!nonneg) {
    if (lp_solver == "gurobi") model$lb = c(rep(-Inf,P), rep(0,N))
    if (lp_solver == "glpk") model$bounds = list(lower=list(ind=1:P,
                                                            val=rep(-Inf,P)))
  }
  
  # Remove nonnegativity constraint on intercepts, if needed
  if (intercept && nonneg) {
    if (lp_solver == "gurobi") model$lb = c(rep(c(-Inf, rep(0,p-1)), r), rep(0,N))
    if (lp_solver == "glpk") model$bounds = list(lower=list(ind=(0:(r-1))*p + 1,
                                                            val=rep(-Inf,r))) 
  }
  
  # Noncrossing constraints, if we're asked to
  if (noncross) {
    # If there's no q0 passed, then just use the training points
    if (is.null(q0)) q0 = qarr
    # If there's one passed, then account for intercept if needed!
    if (!is.null(q0) && intercept) {
      n0 = dim(q0)[1]; a0 = array(NA, dim=c(n0,p,r))
      for (k in 1:r) a0[,,k] = cbind(rep(1,n0), q0[,,k])  
      q0 = a0
    }

    # Now implement the noncrossing constraints
    n0 = dim(q0)[1]
    nrowA = nrow(model$A)
    model$A = rbind(model$A, Matrix(0, nrow=n0*(r-1), ncol=N+P, sparse=TRUE))
    for (k in 1:(r-1)) {
      model$A[nrowA + (k-1)*n0 + 1:n0, (k-1)*p + 1:p] = -q0[,,k]
      model$A[nrowA + (k-1)*n0 + 1:n0, k*p + 1:p] = q0[,,k+1]
      model$sense[nrowA + (k-1)*n0 + 1:n0] = ">="
      model$rhs[nrowA + (k-1)*n0 + 1:n0] = 0
    }
  }

  # Call Gurobi's LP solver, store results
  if (lp_solver == "gurobi") {
    a = gurobi(model=model, params=params)
    alpha = matrix(a$x[1:P],p,r) 
    status = a$status
  }

  # Call GLPK's LP solver, store results
  else if (lp_solver == "glpk") {
    a = Rglpk_solve_LP(obj=model$obj, mat=model$A, dir=model$sense,
                       rhs=model$rhs, bounds=model$bounds, control=params)  
    alpha = matrix(a$solution[1:P],p,r)
    status = a$status
  }

  return(enlist(alpha, status))
}
