package com.zy.alg.service;

import com.zy.alg.util.CategoryCorpusInfo;

import java.util.List;
import java.util.Map;

/**
 * @author zhangycqupt@163.com
 * @date 2018/08/26 21:00
 */
public interface NaiveBayes {
    /**
     * 朴素贝叶斯模型训练器
     *
     * @param trainData       训练语料
     * @param outputModelPath 输出路径
     */
    void modelTrainer(List<CategoryCorpusInfo> trainData, String outputModelPath);

    /**
     * 朴素贝叶斯模型分类器
     *
     * @param title   测试文本主题
     * @param content 测试文本内容
     * @param count   返回类目数
     * @return 识别类目与权值
     */
    List<Map.Entry<String, Double>> modelClassier(String title, String content, int count);
}
