package opt.test;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.TwoColorsEvaluationFunction;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.util.Arrays;

/**
 * @author Daniel Cohen dcohen@gatech.edu
 * @version 1.0
 */
public class TwoColorsTest {
    /** The number of colors */
    private static final int k = 2;
    /** The N value */
    private static final int N = 100*k;

    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, k+1);
        EvaluationFunction ef = new TwoColorsEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 100);
        fit.train();
        System.out.println(ef.value(rhc.getOptimal()));
        
        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        fit = new FixedIterationTrainer(sa, 100);
        fit.train();
        System.out.println(ef.value(sa.getOptimal()));
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
        fit = new FixedIterationTrainer(ga, 100);
        fit.train();
        System.out.println(ef.value(ga.getOptimal()));
        
        MIMIC mimic = new MIMIC(50, 10, pop);
        fit = new FixedIterationTrainer(mimic, 15);
        fit.train();
        System.out.println(ef.value(mimic.getOptimal()));
    }
}
