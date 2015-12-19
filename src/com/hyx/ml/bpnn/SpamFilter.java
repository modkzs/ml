package com.hyx.ml.bpnn;

import com.hyx.ml.feature.Data;
import com.hyx.ml.feature.DataReader;

import java.io.IOException;

/**
 * Created by yixuanhe on 11/14/15.
 */
public class SpamFilter {
    public BPNN bpnn;

    static int HIDDEN_NUMBER = 150;

    public SpamFilter(int featureLen){
        bpnn = new BPNN(featureLen, HIDDEN_NUMBER, 1, new Sigmoid(), 1, 0.5, 0.05, 1);
        bpnn.setStep(1);
    }

    public void train(double[][] wordBag, double[][] tag){
        bpnn.train(wordBag, tag);
    }

    public boolean predict(double[] text){
        double[] v = bpnn.getOutput(text);
        return  (v[0] > 0.5);
    }

    public static void main(String[] args) throws IOException {
        int featureNum = 100;
        Data data = DataReader.dataRead("data/feature_train_100_filter.csv", 800000, featureNum);
        double[][] X = data.X;
        double[][] Y = data.Y;

        SpamFilter sf = new SpamFilter(featureNum);


        int length = X.length;
        int train_len = 750000;

        double[][] x_train = new double[train_len][featureNum];
        double[][] y_train = new double[train_len][featureNum];
        double[][] x_test = new double[length-train_len][featureNum];
        double[][] y_test = new double[length-train_len][featureNum];

        for(int i = 0; i < train_len; i++){
            x_train[i] = X[i];
            y_train[i] = Y[i];
        }

        for(int i = train_len; i < length; i++){
            x_test[i-train_len] = X[i];
            y_test[i-train_len] = Y[i];
        }

        sf.train(x_train, y_train);


        int TT = 0;
        int TF = 0;
        int FT = 0;
        int FF = 0;

//        int right = 0;
//        for (int i = 0; i < length; i++){
//            if (Y[i][0] == 0)
//                right++;
//        }
//        System.out.println(right);

        length = x_test.length;
        for (int i = 0; i < length; i++){
            if (sf.predict(x_test[i]) && y_test[i][0] == 1){
                FF += 1;
            }
            else  if (sf.predict(x_test[i]) && y_test[i][0] == 0){
                FT += 1;
            }
            else if (!sf.predict(x_test[i]) && y_test[i][0] == 1){
                TF += 1;
            }
            else
                TT += 1;
        }

        System.out.println("TT : " + TT);
        System.out.println("TF : " + TF);
        System.out.println("FT : " + FT);
        System.out.println("FF : " + FF);

        double rate = 0.3*(0.65*TT/(TT+TF) + 0.35*TT/(TT+FT)) + 0.7*(0.65*FF/(FF+FT) + 0.35*FF/(FF+TF));
        System.out.print("rate : " + rate);

    }
}
