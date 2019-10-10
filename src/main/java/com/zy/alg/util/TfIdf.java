package com.zy.alg.util;


import com.zbj.alg.seg.domain.Result;
import com.zbj.alg.seg.domain.Term;
import com.zbj.alg.seg.service.ServiceSegModel;
import org.nlpcn.commons.lang.tire.domain.Forest;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author zhangycqupt@163.com
 * @date 2018/08/29 9:32
 */
public class TfIdf {
    /**
     * 计算文档下TF-IDF
     *
     * @param ssme     seg interface
     * @param tagDic   tag dictionary
     * @param tarinMap category corpus
     * @return <category,<word,weight>>
     */
    public static Map<String, Map<String, Double>> compute(ServiceSegModel ssme, Forest tagDic,
                                                           Map<String, List<String>> tarinMap) {
        Map<String, Map<String, Double>> categoryWordWeight = new HashMap<>();
        // 文档总数
        int documentNumber = tarinMap.size();
        Map<String, Map<String, Double>> categoryWordCount = new HashMap<>();
        for (Map.Entry<String, List<String>> q : tarinMap.entrySet()) {
            Map<String, Double> tmpMap = new HashMap<>();
            categoryWordCount.put(q.getKey(), tmpMap);
            for (String text : q.getValue()) {
                Result terms = ssme.parserQueryUser(text.toLowerCase(), "2", tagDic);
                for (Term t : terms) {
                    if (t.getNatureStr().equals("w")
                            || t.toString().split("/").length != 2
                            || t.getName().length() > 10
                            || t.getName().length() < 2
                            || t.getNatureStr().equals("m")) {
                        continue;
                    }
                    if (tmpMap.containsKey(t.getName())) {
                        tmpMap.put(t.getName(), tmpMap.get(t.getName()) + 1.0);
                    } else {
                        tmpMap.put(t.getName(), 1.0);
                    }
                }
            }
        }
        // compute tf-idf
        for (Map.Entry<String, Map<String, Double>> q : categoryWordCount.entrySet()) {
            String category = q.getKey();
            double weight;
            // 当前文档总次数
            int allWordNum = q.getValue().size();
            Map<String, Double> wordWeightMap = new HashMap<>();
            for (Map.Entry<String, Double> qq : q.getValue().entrySet()) {
                String word = qq.getKey();
                double count = qq.getValue();
                double tf = count / allWordNum;
                // 当前word出现在文档中文档数
                double nw = 0.0;
                for (Map<String, Double> w : categoryWordCount.values()) {
                    if (w.containsKey(word)) {
                        nw += 1.0;
                    }
                }
                double idf = Math.log((1 + documentNumber) / (1 + nw));
                weight = tf * idf;
                wordWeightMap.put(word, weight);
            }
            categoryWordWeight.put(category, wordWeightMap);
        }

        return categoryWordWeight;
    }
}
