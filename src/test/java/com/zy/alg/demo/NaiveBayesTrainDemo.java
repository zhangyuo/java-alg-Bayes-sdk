package com.zy.alg.demo;

import com.zy.alg.service.NaiveBayes;
import com.zy.alg.service.NaiveBayesImpl;
import com.zy.alg.util.CategoryCorpusInfo;
import com.zy.alg.util.FileReader;

import java.util.List;

/**
 * @author zhangycqupt@163.com
 * @date 2018/08/26 21:24
 */
public class NaiveBayesTrainDemo {
    public static void main(String[] args) {
        // 输入训练语料路径
        String trainCorpusPath = "G:\\project\\Bayes\\input\\";
        // 输出模型文件路径
        String outPutModelPath = "G:\\project\\Bayes\\output\\";

        // 训练语料
        String trainFile = trainCorpusPath + "TrainCorpus.csv";
        // 读取模型训练语料
        List<CategoryCorpusInfo> trainData = FileReader.getTrainData(trainFile);
        // 类目标签库
        String categoryTagFile = trainCorpusPath + "CategoryTagLibrary";

        // 模型训练
        NaiveBayes naiveBayes = new NaiveBayesImpl(categoryTagFile);
        naiveBayes.modelTrainer(trainData, outPutModelPath);
    }
}
