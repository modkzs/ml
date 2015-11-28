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
        bpnn = new BPNN(featureLen, HIDDEN_NUMBER, 1, new Sigmoid(), 1, 0.5, 0.05);
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
        Data data = DataReader.dataRead("data/sample_train_100_file.csv", 80136);
        double[][] X = data.X;
        double[][] Y = data.Y;

        SpamFilter sf = new SpamFilter(featureNum);
        sf.train(X, Y);

        int length = X.length;

        int TT = 0;
        int TF = 0;
        int FT = 0;
        int FF = 0;

        int right = 0;
        for (int i = 0; i < length; i++){
            if (Y[i][0] == 0)
                right++;
        }
        System.out.println(right);

        for (int i = 0; i < length; i++){
            if (sf.predict(X[i]) && Y[i][0] == 1){
                FF += 1;
            }
            else  if (sf.predict(X[i]) && Y[i][0] == 0){
                FT += 1;
            }
            else if (!sf.predict(X[i]) && Y[i][0] == 1){
                TF += 1;
            }
            else
                TT += 1;
        }

        System.out.println("TT : " + TT);
        System.out.println("TF : " + TF);
        System.out.println("FT : " + FT);
        System.out.println("FF : " + FF);
    }
}
