package opt.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;


/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class StudentScoresTest {
    private static int inputLayer = 5, hiddenLayer = 20, outputLayer = 1, trainingIterations = 5000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = myInitializeInstances();
    private static Instance[] instances = set.getInstances();

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for (int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[]{inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        networks[0] = factory.createClassificationNetwork(
                new int[]{inputLayer, hiddenLayer, outputLayer});
        nnop[0] = new NeuralNetworkOptimizationProblem(set, networks[0], measure);
        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(150, 65, 21, nnop[2]);
        for (int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10, 9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for (int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10, 9);

            results += "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        results += "\nTesting results!\n";
        Instance[] testingInstances = myInitializeTestingInstances().getInstances();
        for (int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            double predicted, actual;
            for (int j = 0; j < testingInstances.length; j++) {
                networks[i].setInputValues(testingInstances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(testingInstances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10, 9);
            results += "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct / (correct + incorrect) * 100) + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for (int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for (int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
        }
    }

    private static DataSet myInitializeInstances() {
        try {
            BufferedReader br = new BufferedReader(new FileReader("src/opt/test/StudentsPerformance_training_whole.csv"));
            String line;
            List<Instance> data = new ArrayList<Instance>();
            Pattern pattern = Pattern.compile("[ ,]+");
            while ((line = br.readLine()) != null) {
                String[] split = pattern.split(line.trim());
                double[] input = new double[split.length];
                for (int i = 0; i < input.length - 1; i++) {
                    input[i] = Double.parseDouble(split[i]);
                }
                Instance instance = new Instance(input);
                instance.setLabel(new Instance(Double.parseDouble(split[input.length - 1])));
                data.add(instance);
            }
            br.close();
            Instance[] instances = (Instance[]) data.toArray(new Instance[0]);
            DataSet set = new DataSet(instances);
            set.setDescription(new DataSetDescription(set));
            return set;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

    }

    private static DataSet myInitializeTestingInstances() {
        try {
            BufferedReader br = new BufferedReader(new FileReader("src/opt/test/StudentsPerformance_testing_average.csv"));
            String line;
            List<Instance> data = new ArrayList<Instance>();
            Pattern pattern = Pattern.compile("[ ,]+");
            while ((line = br.readLine()) != null) {
                String[] split = pattern.split(line.trim());
                double[] input = new double[split.length];
                for (int i = 0; i < input.length - 1; i++) {
                    input[i] = Double.parseDouble(split[i]);
                }
                Instance instance = new Instance(input);
                instance.setLabel(new Instance(Double.parseDouble(split[input.length - 1])));
                data.add(instance);
            }
            br.close();
            Instance[] instances = (Instance[]) data.toArray(new Instance[0]);
            DataSet set = new DataSet(instances);
            set.setDescription(new DataSetDescription(set));
            return set;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
