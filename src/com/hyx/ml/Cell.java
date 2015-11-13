package com.hyx.ml;

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

    ActiveFunction active;

    public Cell(ActiveFunction active, int length, double rate, double threshold){
        this.active = active;
        this.length = length;
        this.rate = rate;
        this.threshold = threshold;

        weight = new double[length];

        for (int i = 0; i < length; i++){
            Random rand = new Random();
            // weight[i] in [-1.0 to 1.0]
            weight[i] = rand.nextDouble()*2-1.0;
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
