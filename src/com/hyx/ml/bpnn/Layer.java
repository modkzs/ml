package com.hyx.ml.bpnn;

/**
 * Created by yixuanhe on 11/13/15.
 */
public class Layer {
    // the cell number
    int number;

    // the cell array
    Cell[] cells;

    // the output
    double[] output;

    // the active situation cell
    boolean[] isActive;

    /*
     * number : cell number
     * length : input feature number
     * active : the active function
     * rate : the learning rate
     */
    public Layer(int number, int length, ActiveFunction active, double rate, double threshold, double xi){
        this.number = number;
        this.cells = new Cell[number];
        this.output = new double[number];
        this.isActive = new boolean[number];

        for (int i = 0; i < number; i++){
            this.cells[i] = new Cell(active, length, rate, threshold, xi);
        }
    }

    public double[] calOutput(double[] value){
        for (int i = 0; i < number; i++){
            output[i] = cells[i].calOutput(value);
        }

        return output;
    }

    public double[] getOutput(){
        return output;
    }

    public void update(double[][] gradients){
        for (int i = 0; i < number; i++){
            cells[i].update(gradients[i]);
        }
    }

    public boolean[] calActive(){
        for (int i = 0; i < number; i++){
            isActive[i] = cells[i].isActive();
        }

        return isActive;
    }

    public boolean[] getActive(){
        return isActive;
    }

    // get the weight of ith cell
    public double[] getWeight(int i){
        return cells[i].getWeight();
    }

    // add epsilon to the jth weight of ith cells in this layer
    public void addWeight(int i, int j, double epsilon){
        cells[i].addEpsilon(j, epsilon);
    }

    // minus epsilon to the jth weight of ith cells in this layer
    public void minusWeight(int i, int j, double epsilon){
        cells[i].minusEpsilon(j, epsilon);
    }
}
