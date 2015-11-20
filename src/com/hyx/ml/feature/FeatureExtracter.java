package com.hyx.ml.feature;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by yixuanhe on 11/16/15.
 */
public class FeatureExtracter {
    Set<String> wordBag;

    public FeatureExtracter(){
        wordBag = new HashSet<>();
    }

    public int featureExtract(String fileName) throws IOException {
        BufferedReader in=new BufferedReader(new FileReader(fileName));

        String line;
        while ((line = in.readLine()) != null){
            String[] word = line.split("\\W+");
            for (String w : word)
                wordBag.add(w);
        }
        in.close();

        return wordBag.size();
    }

    public Data getFeature(String fileName, int featureNum) throws IOException {
        String[] bag = new String[featureNum];
        wordBag.toArray(bag);

        ArrayList<double[]> dataX = new ArrayList<>();
        ArrayList<double[]> dataY = new ArrayList<>();

        BufferedReader in=new BufferedReader(new FileReader(fileName));
        String line;

        int l = 0;

        while ((line = in.readLine()) != null){
            if (l == 4574)
                System.out.println();
            double[] tmpX = new double[featureNum];
            double[] tmpY = new double[2];
            String[] word = line.split("\\W+");
            if (word[0].equals("ham")) {
                tmpY[0] = 0;
                tmpY[1] = 1;
            }else {
                tmpY[0] = 1;
                tmpY[1] = 0;
            }

            int n = word.length;

            for (int i = 0; i < featureNum; i++){
                for (int j = 0; j < n; j++){
                    if (bag[i].equals(word[j])){
                        tmpX[i] = 1;
                    }
                }
            }

            dataX.add(tmpX);
            dataY.add(tmpY);
            l+=1;
        }

        in.close();

        int len = dataX.size();

        double[][] X = new double[len][featureNum];
        double[][] Y = new double[len][featureNum];

        dataX.toArray(X);
        dataY.toArray(Y);

        return new Data(X, Y);
    }
}
