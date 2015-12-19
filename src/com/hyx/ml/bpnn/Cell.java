package com.hyx.ml.bpnn;

import java.util.Random;

/**
 * Created by yixuanhe on 11/13/15.
 */
public class Cell {
    //the weight vector which calculate the net value
    double[] weight;

    //the threshold deciding which this cell is active
    double threshold;

    // the cell output value
    double output;

    // the learning rate
    double rate;

    // the length of weight
    int length;

    // regularization parameter
    double xi;

    ActiveFunction active;

    public Cell(ActiveFunction active, int length, double rate, double threshold, double xi){
        this.active = active;
        this.length = length;
        this.rate = rate;
        this.threshold = threshold;
        this.xi = xi;

        weight = new double[length+1];

        for (int i = 0; i < length; i++){
            Random rand = new Random();
            // weight[i] in [-0.2 to 0.2]
            weight[i] = (rand.nextDouble()*2-1.0)/5;
            //weight[i] = rand.nextDouble()*10;
        }
    }

    public Cell(double[] weight, ActiveFunction active, double rate){
        this.weight = weight;
        this.active = active;
        this.length = weight.length;
        this.rate = rate;
    }

    public double calOutput(double[] value){
        output = active.active(value, weight);
        return output;
    }

    public double getOutput(double[] value){
        return output;
    }

    public boolean isActive(){
        return this.threshold < this.output;
    }

    public void update(double[] gradient){
        for (int i = 0; i < length; i++){
            weight[i] = weight[i] + rate*gradient[i];
            weight[i] = weight[i] * xi;
        }
    }

    public double[] getWeight(){
        return weight;
    }

    //add epsilon to weight i, this is used to test whether the derivative is right
    public void addEpsilon(int i, double epsilon){
        weight[i] += epsilon;
    }

    //minus epsilon to weight i, this is used to test whether the derivative is right
    public void minusEpsilon(int i, double epsilon){
        weight[i] -= epsilon;
    }
}
