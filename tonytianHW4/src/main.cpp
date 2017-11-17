#include <Rcpp.h>
using namespace std;
using namespace Rcpp;

double viterbi(NumericMatrix& obs, NumericVector& ts, double theta, int t, int s, NumericMatrix& delta, IntegerMatrix& phi){
    int n = (int)obs.nrow();
    if (delta(s, t)>0) {
        if (t==0) {
            for (int i = 0; i<n; i++) {
                delta(i, t)=log(1.0/n)+log(obs(i,t));
            }
        }
        else {
            double first=-1.0*(double)__INT_MAX__, second=-1.0*(double)__INT_MAX__;
            int firstid=-1, secondid=-1;
            double p1=log(1.0/n*(1-exp(-(ts[t]-ts[t-1])*theta)));
            double p2=log(1-(double)(n-1)/n*(1-exp(-(ts[t]-ts[t-1])*theta)));
            NumericVector temp(n);
            for (int i = 0; i<n; i++) {
                double v=viterbi(obs, ts, theta, t-1, i, delta, phi);
                if (v+p1>first) {
                    firstid=i;
                    first=v+p1;
                    second=first;
                    secondid=firstid;
                }
                else if (v+p1>second){
                    secondid=i;
                    second=v+p1;
                }
                temp[i]=v+p2;
            }
            for (int i=0; i<n; i++) {
                double tem=temp[i];
                if (i==firstid) {
                    if (second>tem) {
                        delta(i, t)=second;
                        phi(i, t)=secondid;
                    }
                    else {
                        delta(i, t)=tem;
                        phi(i, t)=i;
                    }
                }
                else {
                    if (first>tem) {
                        delta(i, t)=first;
                        phi(i, t)=firstid;
                    }
                    else {
                        delta(i, t)=tem;
                        phi(i, t)=i;
                    }
                }
                delta(i,t)+=log(obs(i,t));
            }
        }
    }
    return delta(s, t);
}


//' Viterbi Algorithm for Symmetric Continuous-time HMM (Rcpp)
//'
//' This function implements a Viterbi algorithm for the continuous-time Hidden Markov Model in a specific condition: n uniform initial states and a special-case dual transition probabilities related to the timestamp and parameter theta (detailed formula in the pdf instruction). The input arguments include the timestamps of each observation represented in a row with size m, a transition parameter theta used for the definition of the transition probabilities, a matrix containing all the conditional probabilities for each observation given each state at that timestamp. Note the time complexity is O(mn).
//'
//' @param ts size m positive float values characterize the timestamps for the observations
//' @param theta a single float value as the tansition parameter determining the transition probabilities
//' @param obs size n*m matrix containing all the conditional probabilities for each observation given each state at that timestamp.
//' @return vector of Viterbi path representing the most likely state at each timestamp with observation
//' @examples
//' obs <- matrix(c(0.88,0.10,0.88,0.10,0.02,0.30,0.02,0.30,0.10,0.60),2,5)
//' theta <- log(2)
//' ts <- c(1,2,3,4,5)
//' ctmcViterbi(ts, theta, obs)
//'
//' ts <- c(1,2.95,3,4,5)
//' ctmcViterbi(ts, theta, obs)
//'
//' @export
// [[Rcpp::export]]
IntegerVector ctmcViterbi(NumericVector ts, double theta, NumericMatrix obs) {
    int m = (int)ts.size();
    int n = (int)obs.nrow();
    if ( obs.ncol() != m )
        stop("The input matrix does not conform to the other parameters");
    NumericMatrix delta(n, m);
    IntegerMatrix phi(n, m);
    fill(delta.begin(), delta.end(), 1);
    IntegerVector viterbiPath(m);
    double v=viterbi(obs, ts, theta, m-1, 0, delta, phi);
    double ml=-1.0*(double)__INT_MAX__;
    for (int i=0; i<n; i++) {
        v=delta(i, m-1);
        if (v>ml) {
            ml=v;
            viterbiPath[m-1]=i;
        }
    }
    for (int i=m-1; i>0; i--) {
        viterbiPath[i-1]=phi(viterbiPath[i], i);
    }
    return viterbiPath;
}


void forward(NumericMatrix& obs, NumericVector& ts, double theta, NumericMatrix& alpha){
    int m = (int)ts.size();
    int n = (int)obs.nrow();
    for (int i=0; i<n; i++) {
        alpha(i, 0)=(double)1.0e3*obs(i,0);
    }
    for (int t=1; t<m; t++) {
        double p1=1.0/n*(1-exp(-(ts[t]-ts[t-1])*theta));
        double p2=1-(double)(n-1)/n*(1-exp(-(ts[t]-ts[t-1])*theta));
        alpha(0, t)=alpha(0, t-1)*p2;
        for (int i=1; i<n; i++) {
            alpha(0, t)+=alpha(i,t-1)*p1;
        }
        alpha(0, t)*=obs(0, t);
        for (int j=1; j<n; j++) {
            alpha(j, t)=((double)alpha(0, t)/obs(0, t)-(p2-p1)*alpha(0, t-1)+(p2-p1)*alpha(j, t-1))*obs(j, t);
        }
        for (int j=0; j<n; j++) {
            alpha(j, t)*=1.1;
        }
    }
}


void backward(NumericMatrix& obs, NumericVector& ts, double theta, NumericMatrix& beta){
    int m = (int)ts.size();
    int n = (int)obs.nrow();
    for (int i=0; i<n; i++) {
        beta(i, m-1)=1.0;
    }
    for (int t=m-2; t>=0; t--) {
        double p1=1.0/n*(1-exp(-(ts[t+1]-ts[t])*theta));
        double p2=1-(double)(n-1)/n*(1-exp(-(ts[t+1]-ts[t])*theta));
        beta(0, t)=beta(0, t+1)*p2*obs(0, t+1);
        for (int i=1; i<n; i++) {
            beta(0, t)+=beta(i,t+1)*p1*obs(i, t+1);
        }
        for (int j=1; j<n; j++) {
            beta(j, t)=beta(0, t)-(p2-p1)*beta(0, t+1)*obs(0, t+1)+(p2-p1)*beta(j, t+1)*obs(j, t+1);
        }
        for (int j=1; j<n; j++) {
            beta(j, t)*=1.1;
        }
    }
}


//' Forward-backward Algorithm for Symmetric Continuous-time HMM (Rcpp)
//'
//' This function implements the forward-backward algorithm for the continuous-time Hidden Markov Model in a specific condition: n uniform initial states and a special-case dual transition probabilities related to the timestamp and parameter theta (detailed formula in the pdf instruction). The input arguments include the timestamps of each observation represented in a row with size m, a transition parameter theta used for the definition of the transition probabilities, a matrix containing all the conditional probabilities for each observation given each state at that timestamp. Note the time complexity is O(mn).
//'
//' @param ts size m positive float values characterize the timestamps for the observations
//' @param theta a single float value as the tansition parameter determining the transition probabilities
//' @param obs size n*m matrix containing all the conditional probabilities for each observation given each state at that timestamp.
//' @return m*n matrix containing all the conditional probabilities of each state at the timestamps given the observations
//' @examples
//' obs <- matrix(c(0.88,0.10,0.88,0.10,0.02,0.30,0.02,0.30,0.10,0.60),2,5)
//' theta <- log(2)
//' ts <- c(1,2,3,4,5)
//' ctmcForwardBackward(ts, theta, obs)
//'
//' ts <- c(1,2.95,3,4,5)
//' ctmcForwardBackward(ts, theta, obs)
//'
//' @export
// [[Rcpp::export]]
NumericMatrix ctmcForwardBackward(NumericVector ts, double theta, NumericMatrix obs) {
    int m = (int)ts.size();
    int n = (int)obs.nrow();
    if ( obs.ncol() != m )
        stop("The input matrix does not conform to the other parameters");
    
    NumericMatrix condProb(n,m);
    NumericMatrix alpha(n, m);
    NumericMatrix beta(n, m);
    forward(obs, ts, theta, alpha);
    backward(obs, ts, theta, beta);
    for (int t=0; t<m; t++) {
        double sum=0;
        for (int i=0; i<n; i++) {
            sum+=alpha(i, t)*beta(i, t);
        }
        for (int i=0; i<n; i++) {
            condProb(i, t)=alpha(i, t)*beta(i, t)/sum;
        }
    }
    return condProb;
}
