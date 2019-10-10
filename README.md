# 朴素贝叶斯分类模型

## 准备工作

1、初始化

    // 读取训练语料 - 根据语料格式自定义
    List<CategoryCorpusInfo> trainData = FileReader.getTrainData(trainFile);

    // 模型接口初始化
    NaiveBayes naiveBayes = new NaiveBayesImpl();

2、jar包：

    <groupId>org.slf4j</groupId>
    <artifactId>slf4j-simple</artifactId>
    <version>1.7.25</version>

    <groupId>org.projectlombok</groupId>
    <artifactId>lombok</artifactId>
    <version>1.16.18</version>

    <groupId>org.nlpcn</groupId>
    <artifactId>nlp-lang</artifactId>
    <version>1.7.7</version>

    分词项目,可使用NLPChina开源分词项目

## 接口说明：

1、朴素贝叶斯模型训练

com.zy.alg.service.NaiveBayes.modelTrainer;

    /**
     * 朴素贝叶斯模型训练器
     *
     * @param trainData       训练语料
     * @param categoryTagFile 类目标签库文件路径
     *                        注意存储格式:(两类)
     *                        (1)number \t category \t attribution \t tag/count;
     *                        (2)number \t category \t tag/count
     * @param outputModelPath
     */
    void modelTrainer(List<CategoryCorpusInfo> trainData, String categoryTagFile, String outputModelPath);
