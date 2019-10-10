package com.zy.alg.demo;


import com.zy.alg.service.NaiveBayes;
import com.zy.alg.service.NaiveBayesImpl;

import java.util.List;
import java.util.Map;

/**
 * @author zhangycqupt@163.com
 * @date 2018/08/29 9:05
 */
public class NaiveBayesModelTest {
    public static void main(String[] args) {
        // 模型路径
        String modelPath = "G:\\project\\Bayes\\output\\";
        String modelFile = modelPath + "ClassifyCategoryModel";
        // 标签库路径
        String tagLibraryPath = "G:\\project\\Bayes\\input\\";
        String tagLibrary = tagLibraryPath + "CategoryTagLibrary";
        NaiveBayes naiveBayes = new NaiveBayesImpl(tagLibrary, modelFile);
        String title = "办公室家具设计";
        String content = "我需要其它或不确定，要求：项目紧急，急需欢乐颂2刘涛办公室家具设计图纸及制作方案，" +
                "最好能生产。要求相似度80以上工期要求：尽快预算范围：1000详谈";
        List<Map.Entry<String, Double>> list = naiveBayes.modelClassier(title, content, 5);
        for (Map.Entry<String, Double> q : list) {
            System.out.println(q.getKey() + "\t" + q.getValue());
        }
    }
}
