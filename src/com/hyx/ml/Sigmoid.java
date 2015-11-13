package com.hyx.ml;

/**
 * Created by yixuanhe on 11/13/15.
 */
public class Sigmoid implements ActiveFunction {

    @Override
    public double active(double[] x, double[] weight) {
        double input = 0;
        int len = weight.length;
        for (int i = 0; i < len; i++) {
            input += x[i]*weight[i];
        }

        return 1.0/(1 + Math.exp(-input));
    }

    /*
     * The derivative of sigmoid, not with weight.
     * You need to calculate the dericative of weight yourself
     */
    @Override
    public double derivative(double[] x, double[] weight) {
        double input = 0;
        int len = weight.length;
        for (int i = 0; i < len; i++) {
            input += x[i]*weight[i];
        }

        double sigmoid = 1.0/(1 + Math.exp(-input));
        return sigmoid*(1-sigmoid);
    }

    public static void main(String[] args){
        Sigmoid sigmoid = new Sigmoid();
        double[] x = {1, 1, 1, 1, 1};
        double[] weight = {1, 1, 1, 1, 2};
        System.out.println(sigmoid.active(x, weight));
        System.out.print(sigmoid.derivative(x, weight));
    }
}
