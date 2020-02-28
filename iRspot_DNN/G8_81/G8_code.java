import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.PrintWriter;
import java.util.Arrays;

/**
 * @author Adam Gibson
 */
public class prac {

    private static Logger log = LoggerFactory.getLogger(CSVExample1232.class);

    @SuppressWarnings("deprecation")
    public static void main(String[] args) throws Exception {

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("G8_81.txt").getFile()));

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 81;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 2;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 1050;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);

        DataSet allData = iterator.next();
        int n = 1;
        int L= 10;
        PrintWriter pr = new PrintWriter("accuracy.txt");
        double acc[] = new double[L];
        double prc[] = new double[L];
        double recall[] = new double[L];
        double f1[] = new double[L];
        double sum = 0;
        final int numInputs = 81;
        int outputNum = 2;
        long seed = 6;

        int k = L;  // L is numeric value like L=5 means 5-Fold
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(allData);     //Apply normalization to the training data
        //  normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

        for (int m = 0; m < n; m++) {
            allData.shuffle();

            // SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.80);  //Use 65% of data for training
            KFoldIterator kf = new KFoldIterator(k, allData);
            for (int i = 0; i < L; i++) {
                DataSet trainingData = kf.next();
                DataSet testData = kf.testFold();


                //     SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.80);  //Use 65% of data for training
                //  allData.shuffle();
        /*   for (int i = 0; i < L; i++) {
               allData.shuffle();
                DataSet trainingData = testAndTrain.getTrain();
                DataSet testData = testAndTrain.getTest();
*/
                //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
             /*   DataNormalization normalizer = new NormalizerStandardize();
                normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
                normalizer.transform(trainingData);     //Apply normalization to the training data
                normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set
*/
                log.info("Build model....");
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .weightInit(WeightInit.XAVIER)
                    .updater(Updater.ADAGRAD)
                    .activation(Activation.TANH)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Nesterovs(0.9))   //momentum 0.9
                    .updater(new Sgd(0.1))
                    .l2(1e-4)

                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(73).build())       // Dropout: 40%
                    .layer(1, new DenseLayer.Builder().nIn(73).nOut(50).build())
                    .layer(2, new DenseLayer.Builder().nIn(50).nOut(38).build())
                    .layer(3, new DenseLayer.Builder().nIn(38).nOut(22).build())
                    .layer(4, new DenseLayer.Builder().nIn(22).nOut(10).build())
                    .layer(5, new OutputLayer.Builder().nIn(10).nOut(outputNum)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build())
                    .pretrain(false).backprop(true)
                    .build();
                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();
                model.setListeners(new ScoreIterationListener(1));

                  /*    UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            int listenerFrequency = 1;
            model.setListeners(new StatsListener(statsStorage, listenerFrequency));
            uiServer.attach(statsStorage);


*/
                //run the model
                for (int j = 0; j < 1000; j++) {
                    model.fit(trainingData);
                    //model.fit(trainingData);
                }
                //evaluate the model on the test set
                Evaluation eval = new Evaluation(2);
                INDArray output = model.output(testData.getFeatures());
                eval.eval(testData.getLabels(), output);
                System.out.println(eval.stats());
                acc[i] = eval.accuracy();
                prc[i] = eval.precision();
                recall[i] = eval.recall();
                f1[i] = eval.f1();
            }
            Arrays.sort(acc);
            pr.println("-------------Accuracy----------------------");
            for (int j=0; j<acc.length; j++)
            {
                sum =sum+acc[j];
            }
            pr.println("Min = " +acc[0]);
            pr.println("Max = " +acc[acc.length-1]);
            double sum0 = Arrays.stream(acc).sum();
            System.out.println("avg = " + sum0 / L);
            pr.println("avg = " + sum0 / L);

            Arrays.sort(prc);
            pr.println("-------------Precision----------------------");
            pr.println("Min = " +prc[0]);
            pr.println("Max = " +prc[prc.length-1]);
            double sum1 = Arrays.stream(prc).sum();
            pr.println("avg1 = " + sum1 / L);



            Arrays.sort(recall);
            pr.println("-------------Recall----------------------");
            pr.println("Min = " +recall[0]);
            pr.println("Max = " +recall[recall.length-1]);
            double sum2 = Arrays.stream(recall).sum();
            pr.println("avg2 = " + sum2 / L);



            Arrays.sort(f1);
            pr.println("---------------F1 score--------------------");
            pr.println("Min = " +f1[0]);
            pr.println("Max = " +f1[f1.length-1]);
            double sum3 = Arrays.stream(f1).sum();
            pr.println("avg3 = " + sum3 / L);

            pr.close();


        }

    }
}
