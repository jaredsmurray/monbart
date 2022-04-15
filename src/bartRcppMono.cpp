#include <Rcpp.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include "rng.h"
#include "tree.h"
#include "info.h"
#include "funs.h"
#include "bd.h"

using namespace Rcpp;

// [[Rcpp::export]]
List bartRcppMono(IntegerVector yobs, NumericVector y_, 
              NumericVector x_, NumericVector xpred_, 
              IntegerVector yobs0, NumericVector y0_, 
              NumericVector x0_, NumericVector xpred0_, 
              int n00,
              List xinfo_list,
              int burn, int nd, int m, 
              double kfac, double offset, double offset0)
{
  RNGScope scope;  
  RNG gen; //this one random number generator is used in all draws

  //double lambda = 1.0; //this one really needs to be set
  //double nu = 3.0;
  //double kfac=2.0; //original is 2.0
  
  Rcout << "\n*****Into bart main\n";
  
  /*****************************************************************************
  /* Read, format y
  *****************************************************************************/
  std::vector<double> y; //storage for y
  std::vector<double> y0; //storage for y
  double miny = INFINITY, maxy = -INFINITY;
  sinfo allys, ally0s;       //sufficient stats for all of y, use to initialize the bart trees.
  
  for(NumericVector::iterator it=y_.begin(); it!=y_.end(); ++it) {
    y.push_back(*it);
    if(*it<miny) miny=*it;
    if(*it>maxy) maxy=*it;
    allys.sy += *it; // sum of y
    allys.sy2 += (*it)*(*it); // sum of y^2
  }
  size_t n = y.size();
  allys.n = n;
  
  miny = -INFINITY; maxy=INFINITY;
  for(NumericVector::iterator it=y0_.begin(); it!=y0_.end(); ++it) {
    y0.push_back(*it);
    if(*it<miny) miny=*it;
    if(*it>maxy) maxy=*it;
    ally0s.sy += *it; // sum of y0
    ally0s.sy2 += (*it)*(*it); // sum of y0^2
  }
  size_t n0 = y0.size();
  ally0s.n = n0;
  
  //double ybar = allys.sy/n; //sample mean
  //double shat = sqrt((allys.sy2-n*ybar*ybar)/(n-1)); //sample standard deviation
  
  /*****************************************************************************
  /* Read, format X, Xpred
  *****************************************************************************/
  //read x   
  //the n*p numbers for x are stored as the p for first obs, then p for second, and so on.
  std::vector<double> x;
  for(NumericVector::iterator it=x_.begin(); it!= x_.end(); ++it) {
    x.push_back(*it);
  }
  size_t p = x.size()/n;
  
  //read x0   
  //the n*p numbers for x0 are stored as the p for first obs, then p for second, and so on.
  std::vector<double> x0;
  for(NumericVector::iterator it=x0_.begin(); it!= x0_.end(); ++it) {
    x0.push_back(*it);
  }
  
  //x for predictions
  dinfo dip; //data information for prediction
  dip.n = 0;
  std::vector<double> xp;     //stored like x
  if(xpred_.size()) {
    for(NumericVector::iterator it=xpred_.begin(); it!=xpred_.end(); ++it) {
       xp.push_back(*it);
    }
    size_t np = xp.size()/p;
    if(xp.size() != np*p) Rcout << "error, wrong number of elements in prediction data set\n";
    if(np) dip.n=np; dip.p=p; dip.x = &xp[0]; dip.y=0; //there are no y's!
  }
  
  Rcout <<"\nburn,nd,number of trees: " << burn << ", " << nd << ", " << m << endl;
  Rcout <<"\nkfac: " << kfac << endl;
  
  //x cutpoints
  xinfo xi;
  
  xi.resize(p);
  for(int i=0; i<p; ++i) {
    NumericVector tmp = xinfo_list[i];
    std::vector<double> tmp2;
    for(size_t j=0; j<tmp.size(); ++j) {
      tmp2.push_back(tmp[j]);
    }
    xi[i] = tmp2;
  }
  
  //prxi(xi);
  
  
  //size_t nc=100; //100 equally spaced cutpoints from min to max.
  //makexinfo(p,n,&x[0],xi,nc);
  
  /*****************************************************************************
  /* Setup for MCMC
  *****************************************************************************/
  //--------------------------------------------------
  //trees
  std::vector<tree> t(m);
  for(size_t i=0;i<m;i++) t[i].setm(0); //if you sum the fit over the trees you get the fit.
  std::vector<tree> t0(m);
  for(size_t i=0;i<m;i++) t0[i].setm(0);
  //--------------------------------------------------
  //prior and mcmc
  pinfo pi;
  pi.pbd = 1.0; //prob of birth/death move
  pi.pb = .5; //prob of birth given  birth/death
  
  pi.alpha = .95; //prior prob a bot node splits is alpha/(1+d)^beta, d is depth of node
  pi.beta = 2.0; //2 for bart means it is harder to build big trees.
  pi.tau = 3.0/(kfac*sqrt((double)m)); //sigma_mu
  pi.sigma = 1.0; //shat;
  
  Rcout << "\nalpha, beta: " << pi.alpha << ", " << pi.beta << endl;
  Rcout << "sigma, tau: " << pi.sigma << ", " << pi.tau << endl;
  
  //--------------------------------------------------
  //dinfo
  double* allfit = new double[n]; //sum of fit of all trees
  for(size_t i=0;i<n;i++) allfit[i] = offset;
  double* r = new double[n]; //y-(allfit-ftemp) = y-allfit+ftemp
  double* ftemp = new double[n]; //fit of current tree
  dinfo di;
  di.n=n; di.p=p; di.x = &x[0]; di.y=r; //the y for each draw will be the residual 
  //--------------------------------------------------
  //dinfo0
  double* allfit0 = new double[n0]; //sum of fit of all trees
  for(size_t i=0;i<n0;i++) allfit0[i] = offset0;
  double* r0 = new double[n0]; //y-(allfit-ftemp) = y-allfit+ftemp
  double* ftemp0 = new double[n0]; //fit of current tree
  dinfo di0;
  di0.n=n0; di0.p=p; di0.x = &x0[0]; di0.y=r0; //the y for each draw will be the residual 
  
  Rcout << "dinfo full\n";
  
  //--------------------------------------------------
  //storage for ouput
  //in sample fit
  double* pmean = new double[n]; //posterior mean of in-sample fit, sum draws,then divide
  for(size_t i=0;i<n;i++) pmean[i] = 0.0;
  double* pmean0 = new double[n0]; //posterior mean of in-sample fit, sum draws,then divide
  for(size_t i=0;i<n0;i++) pmean0[i] = 0.0;
  
  //out of sample fit
  double* ppredmean=0; //posterior mean for prediction
  double* ppredmean0=0; //posterior mean for prediction
  double* fpredtemp=0; //temporary fit vector to compute prediction
  if(dip.n) {
    ppredmean = new double[dip.n];
    ppredmean0 = new double[dip.n];
    fpredtemp = new double[dip.n];
    for(size_t i=0;i<dip.n;i++) { ppredmean[i]=0.0; ppredmean0[i]=0.0; }
  }
  //for sigma draw
  double rss, restemp;
  
  NumericVector ssigma(nd);
  NumericMatrix sfit(nd,n);
  NumericMatrix spred2(nd,dip.n);
  NumericMatrix sfit0(nd,n0);
  NumericMatrix spred20(nd,dip.n);

  /*****************************************************************************
  /* MCMC
  *****************************************************************************/
  Rcout << "\nMCMC:\n";
  time_t tp;
  int time1 = time(&tp);
  
  for(size_t i=0;i<(nd+burn);i++) {
    if(i%50==0) {
      Rcout << "i: " << i << " sigma: "<< pi.sigma << endl;
    }
    //draw trees
    for(size_t j=0;j<m;j++) {
       fit(t[j],xi,di,ftemp);
       for(size_t k=0;k<n;k++) {
          allfit[k] = allfit[k] - ftemp[k];
          r[k] = y[k]-allfit[k];
       }
       bd(t[j],xi,di,pi,gen);
       drmu(t[j],xi,di,pi,gen);
       fit(t[j],xi,di,ftemp);
       for(size_t k=0;k<n;k++) allfit[k] += ftemp[k];
    }
    for(size_t j=0;j<m;j++) {
       fit(t0[j],xi,di0,ftemp0);
       for(size_t k=0;k<n0;k++) {
          allfit0[k] = allfit0[k] - ftemp0[k];
          r0[k] = y0[k]-allfit0[k];
       }
       bd(t0[j],xi,di0,pi,gen);
       drmu(t0[j],xi,di0,pi,gen);
       fit(t0[j],xi,di0,ftemp0);
       for(size_t k=0;k<n0;k++) allfit0[k] += ftemp0[k];
    }
    
    for(int k=0; k<n00; ++k) {
      double pr0 = R::pnorm(allfit0[k], offset0, 1.0, 1, 0); 
      double pr  = R::pnorm(allfit[k],  offset,  1.0, 1, 0);
      int ty0 = R::runif(0.0, 1.0) < pr0 ? 1 : 0;
      int ty  = R::runif(0.0, 1.0) < pr  ? 1 : 0;
      while(ty0*ty>0) {
        ty0 = R::runif(0.0, 1.0) < pr0 ? 1 : 0;
        ty  = R::runif(0.0, 1.0) < pr  ? 1 : 0;
      }
      //if(i>burn) {
        yobs0[k] = ty0;
        yobs[k]  = ty;
      //}
    }
    
    for(int k=0; k<n; ++k) {
      if(yobs[k]>0) {
        y[k] = rtnormlo1(allfit[k], offset); //correct because allfit has offset inside of it
      } else {
        y[k] = rtnormhi1(allfit[k], offset); //correct because allfit has offset inside of it
      }
      //Rcout << y[k] << '\n';
    }
    for(int k=0; k<n0; ++k) {
      if(yobs0[k]>0) {
        y0[k] = rtnormlo1(allfit0[k], offset0); //rtnorm_1(allfit[k], 1.0, -offset, INFINITY);
      } else {
        y0[k] = rtnormhi1(allfit0[k], offset0); //rtnorm_1(allfit[k], 1.0, -80., -offset);
      }
    }
    
    if(i>=burn) {
      ssigma(i-burn) = pi.sigma;
       for(size_t k=0;k<n;k++) {
         pmean[k] += allfit[k];
         sfit(i-burn, k) = allfit[k];
       }
       if(dip.n) {
         for(size_t k=0;k<dip.n;k++) {
           spred2(i-burn, k) = fit_i(k, t, xi, dip); //tested good
         }
       }
       for(size_t k=0;k<n0;k++) {
         pmean0[k] += allfit0[k];
         sfit0(i-burn, k) = allfit0[k];
       }
       if(dip.n) {
         for(size_t k=0;k<dip.n;k++) {
           spred20(i-burn, k) = fit_i(k, t0, xi, dip); //tested good
         }
       }
    }
  }
  int time2 = time(&tp);
  Rcout << "time for loop: " << time2 - time1 << endl;

  NumericVector bfit(n);
  for(size_t i=0;i<n;i++) bfit[i] = pmean[i]/ (double)nd;
  
  NumericVector bpred(dip.n);
  for(size_t i=0;i<dip.n;i++) bpred[i] = ppredmean[i]/ (double)nd;

  t.clear(); t0.clear();
  delete[] allfit;
  delete[] allfit0;
  delete[] r;
  delete[] r0;
  delete[] ftemp;
  delete[] ftemp0;
  delete[] pmean;
  delete[] ppredmean;
  delete[] fpredtemp;

  return(List::create(_["insample"] = bfit, _["pred"] = bpred, _["sigma"] = ssigma,
                      _["postfit"] = sfit, //_["postpred"] = spred, 
                      _["postpred"] = spred2,
                      _["postfit0"] = sfit0, //_["postpred"] = spred, 
                      _["postpred0"] = spred20, _["yobs"]=yobs, _["yobs0"]=yobs0
                      ));
}
