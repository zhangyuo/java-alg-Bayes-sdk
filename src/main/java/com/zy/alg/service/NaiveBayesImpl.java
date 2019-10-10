package com.zy.alg.service;

import com.zbj.alg.seg.domain.Result;
import com.zbj.alg.seg.domain.Term;
import com.zbj.alg.seg.service.ServiceSegModel;
import com.zbj.alg.seg.service.ServiceSegModelEnhance;
import com.zy.alg.util.CategoryCorpusInfo;
import com.zy.alg.util.Entry;
import com.zy.alg.util.TfIdf;
import org.nlpcn.commons.lang.tire.domain.Forest;
import org.nlpcn.commons.lang.tire.domain.Value;
import org.nlpcn.commons.lang.tire.library.Library;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * @author zhangycqupt@163.com
 * @date 2018/08/26 21:07
 * 注明：
 * 类目标签库文件，需注意存储格式:(两类)
 * (1)number \t category \t attribution \t tag/count;
 * (2)number \t category \t tag/count
 */
public class NaiveBayesImpl implements NaiveBayes {
    private static Logger logger = LoggerFactory.getLogger(NaiveBayesImpl.class);
    private String modelName = "ClassifyCategoryModel";
    /**
     * 默认属性设置
     */
    private static final String ATTR = "属性";
    /**
     * 类目标签库词典
     */
    Forest tagDic = new Forest();
    /**
     * 分词项目加载
     */
    ServiceSegModel ssme;
    /**
     * 类目标签库——<类目，<属性，<标签>>>
     */
    Map<String, Map<String, Set<String>>> categoryTagLib;

    public NaiveBayesImpl(String categoryTagFile) {
        try {
            ssme = ServiceSegModelEnhance.getInstance();
            logger.info("load seg project success");
        } catch (IOException e) {
            logger.error("load seg project failed", e);
        }
        // 加载类目标签库——<类目，<属性，<标签>>>
        categoryTagLib = loadTagLibrary(categoryTagFile);
    }

    /**
     * 类目概率
     */
    Map<String, Double> catePro = new LinkedHashMap<>();
    /**
     * 类目-标签-概率
     */
    Map<String, Map<String, Entry<Double, Double>>> cateWordPro = new HashMap<>();

    /**
     * @param categoryTagFile 类目标签库
     * @param modelFile       模型文件路径
     */
    public NaiveBayesImpl(String categoryTagFile, String modelFile) {
        try {
            ssme = ServiceSegModelEnhance.getInstance();
            logger.info("load seg project success");
        } catch (IOException e) {
            logger.error("load seg project failed", e);
        }
        // 加载类目标签库——<类目，<属性，<标签>>>
        categoryTagLib = loadTagLibrary(categoryTagFile);
        loadModel(modelFile);
    }

    /**
     * 加载分类模型
     *
     * @param modelFile
     */
    private void loadModel(String modelFile) {
        if (modelFile == null
                || modelFile == "") {
            throw new RuntimeException("model file is not exist");
        }
        BufferedReader br;
        try {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(modelFile), "utf-8"));
            String line;
            Boolean categoryFlag = false;
            Boolean categoryWordFalg = false;
            while ((line = br.readLine()) != null) {
                if (line.equals("P(C)")) {
                    categoryFlag = true;
                } else if (line.equals("P(q(k)|C)")) {
                    categoryFlag = false;
                    categoryWordFalg = true;
                }
                if (categoryFlag) {
                    String[] seg = line.split("\t");
                    if (seg.length == 2) {
                        String category = seg[0];
                        double score = Double.parseDouble(seg[1]);
                        catePro.put(category, score);
                    }
                } else if (categoryWordFalg) {
                    String[] seg = line.split("\t");
                    if (seg.length == 4) {
                        String category = seg[0];
                        String tag = seg[1];
                        double score = Double.parseDouble(seg[2]);
                        double tfIdf = Double.parseDouble(seg[3]);
                        Entry entry = new Entry(score, tfIdf);
                        if (cateWordPro.containsKey(category)) {
                            Map<String, Entry<Double, Double>> tmpMap = cateWordPro.get(category);
                            tmpMap.put(tag, entry);
                            cateWordPro.put(category, tmpMap);
                        } else {
                            Map<String, Entry<Double, Double>> tmpMap = new HashMap<>();
                            tmpMap.put(tag, entry);
                            cateWordPro.put(category, tmpMap);
                        }
                    }
                }
            }
            br.close();
            logger.info("load model file success");
        } catch (IOException e) {
            logger.error("load model file failed.", e);
        }
    }

    @Override
    public void modelTrainer(List<CategoryCorpusInfo> trainData, String outputModelPath) {
        // 输出路径初始化
        initOutput(outputModelPath);
        // 模型参数
        Map<String, Map<String, Double>> categoryWordMap = new HashMap<>();
        Map<String, Double> categoryMap = new HashMap<>();
        Map<String, Double> wordMap = new HashMap<>();
        // 读取训练语料并计算模型参数
        Map<String, List<String>> trainMap = readTrainData(trainData, categoryWordMap, categoryMap,
                wordMap, categoryTagLib);
        // compute tf-idf
        Map<String, Map<String, Double>> categoryWordWeight = TfIdf.compute(ssme, tagDic, trainMap);
        // sort and model output
        outputModel(outputModelPath, wordMap, categoryMap, categoryWordMap, categoryWordWeight);
        logger.info("the main pocess of model trainer is over");
    }

    @Override
    public List<Map.Entry<String, Double>> modelClassier(String title, String content, int count) {
        List<Map.Entry<String, Double>> list = new ArrayList<>();
        if ((title == null || title == "")
                && (content == null || content == "")) {
            int num = 0;
            for (Map.Entry<String, Double> q : catePro.entrySet()) {
                num++;
                if (num < count) {
                    list.add(q);
                } else {
                    break;
                }
            }
            return list;
        }
        // title
        Map<String, Double> categoryScoreMap = new HashMap<>();
        Result terms = ssme.parserQueryUser(title.toLowerCase(), "2", tagDic);
        calculateScore(terms, categoryScoreMap);

        //content
        Map<String, Double> categoryScoreMap1 = new HashMap<>();
        Result terms1 = ssme.parserQueryUser(content.toLowerCase(), "2", tagDic);
        calculateScore(terms1, categoryScoreMap1);

        // 加权计算
        Map<String, Double> finalCategoryMap = new HashMap<>();
        double finalScore;
        for (Map.Entry<String, Double> q : categoryScoreMap.entrySet()) {
            String category = q.getKey();
            double titleScore = 0.3 * q.getValue();
            double contentScore = 0.7 * categoryScoreMap1.get(category);
            double categoryScore = catePro.get(category);
            finalScore = (titleScore + contentScore) * categoryScore;
            finalCategoryMap.put(category, finalScore);
        }

        // 排序输出
        Map<String, Double> sortFinalCategoryMap = new LinkedHashMap<>();
        finalCategoryMap.entrySet().stream()
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                .forEachOrdered(e -> sortFinalCategoryMap.put(e.getKey(), e.getValue()));
        int num = 0;
        for (Map.Entry<String, Double> q : finalCategoryMap.entrySet()) {
            num++;
            if (num <= count) {
                list.add(q);
            } else {
                break;
            }
        }

        return list;
    }

    /**
     * 计算类目下标签得分
     *
     * @param terms
     * @param categoryScoreMap
     */
    private void calculateScore(Result terms, Map<String, Double> categoryScoreMap) {
        for (Map.Entry<String, Double> q : catePro.entrySet()) {
            categoryScoreMap.put(q.getKey(), 0.0);
        }
        for (Term t : terms) {
            if (t.getNatureStr().equals("w")
                    || t.toString().split("/").length != 2
                    || t.getName().length() > 10
                    || t.getName().length() < 2
                    || t.getNatureStr().equals("m")) {
                continue;
            }
            // 计算类目下标签分数
            for (Map.Entry<String, Map<String, Entry<Double, Double>>> q : cateWordPro.entrySet()) {
                String category = q.getKey();
                if (q.getValue().containsKey(t.getName())) {
                    double score = cateWordPro.get(category).get(t.getName()).getKey();
                    double tfIdf = cateWordPro.get(category).get(t.getName()).getValue();
                    // tfIdf * score or tfIdf + score(最大最小归一化)
                    categoryScoreMap.put(category, categoryScoreMap.get(category) + tfIdf * score);
                }
            }
        }
    }

    /**
     * 模型打印输出
     *
     * @param outputModelPath
     * @param wordMap
     * @param categoryMap
     * @param categoryWordMap
     * @param categoryWordWeight
     */
    private void outputModel(String outputModelPath, Map<String, Double> wordMap, Map<String, Double> categoryMap,
                             Map<String, Map<String, Double>> categoryWordMap, Map<String, Map<String, Double>> categoryWordWeight) {
        String outputModelFile = outputModelPath;
        if (!outputModelPath.endsWith(File.separator)) {
            outputModelFile = outputModelPath + modelName;
        }
        String outputModel = outputModelFile + modelName;
        PrintWriter pw;
        try {
            pw = new PrintWriter(new OutputStreamWriter(
                    new FileOutputStream(outputModel), "utf-8"), true);
            // sort wordMap
            Map<String, Double> sortWordMap;
            if (!wordMap.isEmpty()) {
                sortWordMap = new LinkedHashMap<>();
                wordMap.entrySet().stream()
                        .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                        .forEachOrdered(e -> sortWordMap.put(e.getKey(), e.getValue()));
                pw.println("Count(Word)");
                for (Map.Entry<String, Double> t : sortWordMap.entrySet()) {
                    if (t.getValue() >= 2) {
                        pw.println(t.getKey() + "##" + t.getValue());
                    }
                }
            } else {
                logger.warn("wordMap is empty");
            }
            // sort category map
            Map<String, Double> sortCategoryMap;
            if (!categoryMap.isEmpty()) {
                sortCategoryMap = new LinkedHashMap<>();
                categoryMap.entrySet().stream()
                        .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                        .forEachOrdered(e -> sortCategoryMap.put(e.getKey(), e.getValue()));
                Double sum = 0.0;
                for (Map.Entry<String, Double> t : sortCategoryMap.entrySet()) {
                    if (t.getValue() > 10000) {
                        sum += 10000 + 0.05 * (t.getValue() - 10000);
                    } else {
                        sum += t.getValue();
                    }
                }
                pw.println("P(C)");
                for (Map.Entry<String, Double> t : sortCategoryMap.entrySet()) {
                    if (t.getValue() > 10000) {
                        double score = Math.log((10000 + 0.05 * (t.getValue() - 10000)) / sum);
                        // 保证正数
                        score = Math.exp(score);
                        pw.println(t.getKey() + "\t" + score);
                    } else {
                        double score = Math.log(t.getValue() / sum);
                        // 保证正数
                        score = Math.exp(score);
                        pw.println(t.getKey() + "\t" + score);
                    }
                }
            } else {
                logger.warn("categoryMap is empty");
            }
            // sort category word map
            pw.println("P(q(k)|C)");
            for (Map.Entry<String, Map<String, Double>> t : categoryWordMap.entrySet()) {
                Map<String, Double> sortWordMap1;
                if (!t.getValue().isEmpty()) {
                    sortWordMap1 = new LinkedHashMap<>();
                    t.getValue().entrySet().stream()
                            .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                            .forEachOrdered(e -> sortWordMap1.put(e.getKey(), e.getValue()));
                    double num = 0;
                    for (Map.Entry<String, Double> tt : sortWordMap1.entrySet()) {
                        num += tt.getValue();
                    }
                    for (Map.Entry<String, Double> tt : sortWordMap1.entrySet()) {
                        double value = Math.log(tt.getValue() / num);
                        double tfIdf = categoryWordWeight.get(t.getKey()).get(tt.getKey());
                        if (Math.log(tt.getValue() / num) <= -20) {
                            pw.println(t.getKey() + "\t" + tt.getKey() + "\t" + 0.01 + "\t" + tfIdf);
                            continue;
                        }
                        pw.println(t.getKey() + "\t" + tt.getKey() + "\t" + (value + 20.0) + "\t" + tfIdf);
                    }
                } else {
                    logger.warn("wordMap in category " + t.getKey() + " is empty");
                }
            }
            pw.close();
            logger.info("output model success and path is " + outputModel);
        } catch (IOException e) {
            logger.error("output model failed.", e);
        }
    }

    /**
     * 输出路径初始化
     *
     * @param outputModelPath
     */
    private void initOutput(String outputModelPath) {
        if (outputModelPath == null || outputModelPath == "") {
            throw new RuntimeException("输出路径为空");
        }
        String outputPath = outputModelPath + File.separator;
        Path outputModelRootPath = Paths.get(outputPath);
        if (!Files.exists(outputModelRootPath)) {
            try {
                Files.createDirectories(outputModelRootPath);
                logger.info("Creating output path success");
            } catch (IOException e) {
                logger.error("Creating output path failed!", e);
            }
        }
    }

    /**
     * 读取训练语料
     *
     * @param trainData
     * @param categoryWordMap
     * @param wordMap
     * @param categoryTagLib
     */
    private Map<String, List<String>> readTrainData(List<CategoryCorpusInfo> trainData, Map<String,
            Map<String, Double>> categoryWordMap,
                                                    Map<String, Double> categoryMap, Map<String, Double> wordMap,
                                                    Map<String, Map<String, Set<String>>> categoryTagLib) {
        if (trainData.isEmpty()) {
            throw new RuntimeException("train data is empty and the trainer can not continue");
        }
        Map<String, List<String>> trainMap = new HashMap<>();
        for (CategoryCorpusInfo q : trainData) {
            String category = q.getCategory();
            String content = q.getContent();
            if (category == null
                    || content == null
                    || category == ""
                    || content == "") {
                continue;
            }
            // 获取category map - 类目计数
            if (categoryMap.containsKey(category)) {
                // 类别均衡限制
                if (categoryMap.get(category) < 100000) {
                    categoryMap.put(category, categoryMap.get(category) + 1.0);
                }
                // 分类存储语料
                List<String> list = trainMap.get(category);
                list.add(content);
                trainMap.put(category, list);
            } else {
                categoryMap.put(category, 1.0);
                // 分类存储语料
                List<String> list = new ArrayList<>();
                list.add(content);
                trainMap.put(category, list);
            }
            Result terms = ssme.parserQueryUser(content.toLowerCase(), "2", tagDic);
            for (Term t : terms) {
                if (t.getNatureStr().equals("w")
                        || t.toString().split("/").length != 2
                        || t.getName().length() > 10
                        || t.getName().length() < 2
                        || t.getNatureStr().equals("m")) {
                    continue;
                }
                // 获取word map - 词频
                if (wordMap.containsKey(t.getName())) {
                    wordMap.put(t.getName(), wordMap.get(t.getName()) + 1.0);
                } else {
                    wordMap.put(t.getName(), 1.0);
                }
                // 获取category word map
                if (!categoryWordMap.containsKey(category)) {
                    Map<String, Double> temMap = new HashMap<>();
                    double score = calculateScore(category, t, categoryTagLib);
                    temMap.put(t.getName(), score);
                    categoryWordMap.put(category, temMap);
                } else {
                    if (categoryWordMap.get(category).containsKey(t.getName())) {
                        double score = calculateScore(category, t, categoryTagLib);
                        categoryWordMap.get(category).put(t.getName(),
                                categoryWordMap.get(category).get(t.getName()) + score);
                    } else {
                        double score = calculateScore(category, t, categoryTagLib);
                        categoryWordMap.get(category).put(t.getName(), score);
                    }
                }
            }
        }
        logger.info("read train data finished");
        return trainMap;
    }

    private double calculateScore(String category, Term t,
                                  Map<String, Map<String, Set<String>>> categoryTagLib) {
        double score = 2.0;
        // 规则1
        if (t.getName().length() > 1
                && !t.getRealName().matches("[a-z]*")) {
            score += 1;
        }
        // 规则2
        if (t.getNatureStr().equals("n")) {
            score += 1;
        }
        // 规则3-类目-属性
        if (categoryTagLib.containsKey(category)) {
            // 无属性情形
            if (categoryTagLib.get(category).containsKey(ATTR)) {
                for (Map.Entry<String, Set<String>> q : categoryTagLib.get(category).entrySet()) {
                    if (q.getValue().contains(t.getName())) {
                        score += 10;
                    }
                }
            } else {

            }
        }
        // 规则4
        if (t.getName().equals(category)) {
            score += 20;
        }
        // 词向量-待添加

        return score;
    }

    /**
     * 读取类目标签库
     *
     * @param categoryTagFile
     * @return
     */
    private Map<String, Map<String, Set<String>>> loadTagLibrary(String categoryTagFile) {
        if (categoryTagFile == null || categoryTagFile == "") {
            throw new RuntimeException("the path of input category tag library is illegal");
        }
        Map<String, Map<String, Set<String>>> categoryTagLib = new HashMap<>();
        BufferedReader br;
        try {
            br = new BufferedReader(new InputStreamReader(
                    new FileInputStream(categoryTagFile), "utf-8"));
            String line;
            while ((line = br.readLine()) != null) {
                String[] seg = line.split("\t");
                String attribuition = ATTR;
                if (seg.length >= 3) {
                    if (!seg[2].contains("/")) {
                        // 含属性词
                        attribuition = seg[2];
                    }
                    String category = seg[1];
                    Map<String, Set<String>> tmpMap = categoryTagLib.get(category);
                    if (tmpMap == null) {
                        tmpMap = new HashMap<>();
                        categoryTagLib.put(category, tmpMap);
                    }
                    Set<String> set = tmpMap.get(attribuition);
                    if (set == null) {
                        set = new HashSet<>();
                        tmpMap.put(attribuition, set);
                    }
                    for (int i = 3; i < seg.length; i++) {
                        String[] seg1 = seg[i].split("/");
                        String tag = seg1[0];
                        set.add(tag);
                        // 插入词典 - 未使用标签库词频
                        Library.insertWord(tagDic, new Value(tag, "UserDefine", "2000"));
                    }
                }
            }
            br.close();
            logger.info("reading category tag lirbrary success");
        } catch (IOException e) {
            logger.error("reading category tag lirbrary failed", e);
        }
        return categoryTagLib;
    }
}
