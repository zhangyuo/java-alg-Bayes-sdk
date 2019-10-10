package com.zy.alg.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * @author zhangycqupt@163.com
 * @date 2018/08/26 21:37
 */
public class FileReader {
    private static Logger logger = LoggerFactory.getLogger(FileReader.class);

    /**
     * 获取训练语料数据
     *
     * @param trainFile
     * @return 格式化训练语料
     */
    public static List<CategoryCorpusInfo> getTrainData(String trainFile) {
        if (trainFile == null || trainFile == "") {
            return null;
        }
        List<CategoryCorpusInfo> list = new ArrayList<>();
        BufferedReader br;
        try {
            br = new BufferedReader(new InputStreamReader(
                    new FileInputStream(trainFile), "utf-8"));
            String line;
            int num = 0;
            while ((line = br.readLine()) != null){
                num++;
                if(num == 1) {
                    continue;
                } else {
                    String[] seg = line.split(",");
                    if (seg.length == 9){
                        String category = seg[1]+"&"+seg[3]+"&"+seg[5];
                        String content = seg[7]+"。"+seg[8];
                        CategoryCorpusInfo categoryCorpusInfo = new CategoryCorpusInfo();
                        categoryCorpusInfo.setCategory(category);
                        categoryCorpusInfo.setContent(content);
                        list.add(categoryCorpusInfo);
                    }
                }
            }
            br.close();
        } catch (IOException e) {
            logger.error("reading train file failed", e);
        }
        return list;
    }
}
