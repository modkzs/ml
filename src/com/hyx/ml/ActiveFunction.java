package com.hyx.ml;

/**
 * Created by yixuanhe on 11/13/15.
 */
public interface ActiveFunction {
    // the active function
    double active(double[] x, double[] weight);
    // the derivative of active function
    double derivative(double[] x, double[] weight);
}
