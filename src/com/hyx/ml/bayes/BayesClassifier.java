package com.hyx.ml.bayes;

import com.hyx.ml.feature.Data2;
import com.hyx.ml.feature.DataReader;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by yixuanhe on 11/20/15.
 */
public class BayesClassifier {
    // the class nuber
    int number;

    // the probability of each item in each
    private Map<Integer, Double>[] itemProb;

    // the total weight value of spam and ham in word vector
    private double totalItemNum[];

    //the total weight value of each item class
    private Map<Integer, Double>[] itemNum;

    public BayesClassifier(int n){
        number = n;
        totalItemNum = new double[n];

        itemProb = new HashMap[n];
        itemNum = new HashMap[n];

        for (int i = 0; i < n; i++){
            itemProb[i] = new HashMap<>();
            itemNum[i] = new HashMap<>();
        }
    }

    public void train(double[][] X, int[][] Y){
        int leng = X.length;
        for (int i = 0; i < leng; i++){
            int label = Y[i][0];
            double[] xs = X[i];
            Map<Integer, Double> tmpItemNum = itemNum[label];
            int lenX = xs.length;
            for (int j = 0; j < lenX; j++){
                double x = xs[j];
                totalItemNum[label] += x;
                if (tmpItemNum.containsKey(j)){
                    double n = tmpItemNum.get(j);
                    n = n + x;
                    tmpItemNum.replace(j, n);
                } else {
                    tmpItemNum.put(j, x);
                }
            }
        }

        for (int i = 0; i < number; i++){
            Map<Integer, Double> tmpItemNum = itemNum[i];
            for (Map.Entry<Integer, Double> item : tmpItemNum.entrySet()){
                itemProb[i].put(item.getKey(), item.getValue()/totalItemNum[i]);
            }
        }

    }

    public int predict(double[] X){
        int len = X.length;

        double prob[] = new double[number];
        for (int i = 0; i < number; i++)
            prob[i] = 1;

        for (int i = 0; i < len; i++){
            for (int j = 0; j < number; j++)
                prob[j] = prob[j] * X[i] * itemProb[j].get(i);
        }

        int result = 0;

        for (int i = 1; i < number; i++){
            if (prob[i] > prob[result])
                result = i;
        }

        return result;
    }

    /*
     * test train data, return number* number matrix
     * the Mij in this matrix means that the number of
     * item which belongs to class i but be judged to class j
     */

    public int[][] test(double[][] X, int[][] Y){
        int[][] result = new int[number][number];
        int len = X.length;

        for (int i = 0; i < len; i++){
            int judge_Y  = predict(X[i]);
            result[Y[i][0]][judge_Y] += 1;
        }

        return result;
    }

    public static void main(String[] args) throws IOException {
        Data2 data = DataReader.intDataRead("data/sample_train_300_file.csv", 80176);
        BayesClassifier bayes = new BayesClassifier(2);
        bayes.train(data.X, data.Y);
        //data = DataReader.intDataRead("data/sample_test_100_file.csv", 20088);
        int[][] result = bayes.test(data.X, data.Y);

        for (int[] rs : result){
            for (int r : rs){
                System.out.print(r + " ");
            }
            System.out.println();
        }
    }
}
