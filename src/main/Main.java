package main;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import process.DataReader;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.WordsFromFile;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;
import domain.Document;
import model.EnsembleLearner;

public class Main {

	public static void main(String args[]) throws Exception {

		List<String> projects = new ArrayList<String>();

		projects.add("argouml");
		projects.add("columba-1.4-src");
		projects.add("hibernate-distribution-3.3.2.GA");
		projects.add("jEdit-4.2");
		projects.add("jfreechart-1.0.19");
		projects.add("apache-jmeter-2.10");
		projects.add("jruby-1.4.0");
		projects.add("sql12");
		
		
		double ratio = 0.1;

		for (int target = 0; target < projects.size(); target++) {

			System.out.println("targe project: " + projects.get(target));

			EnsembleLearner eLearner = new EnsembleLearner();
			List<Document> comments = DataReader.readComments("data/");	
			Set<String> projectForTesting = new HashSet<String>();
			projectForTesting.add(projects.get(target));

			List<Document> testDoc = DataReader.selectProject(comments, projectForTesting);

			for (int source = 0; source < projects.size(); source++) {
				if (source == target)
					continue;

				Set<String> projectForTraining = new HashSet<String>();
				projectForTraining.add(projects.get(source));

				List<Document> trainDoc = DataReader.selectProject(comments, projectForTraining);

				// System.out.println("building dataset for training");
				String trainingDataPath = "F:/research/technical debt/trainingData.arff";
				DataReader.outputArffData(trainDoc, trainingDataPath);

				// System.out.println("building dataset for testing");
				String testingDataPath = "F:/research/technical debt/testingData.arff";
				DataReader.outputArffData(testDoc, testingDataPath);

				if (eLearner.getTestData() == null) {
					Instances tmp = DataSource.read(testingDataPath);
					tmp.setClassIndex(1);
					eLearner = new EnsembleLearner(tmp);
				}

				// string to word vector (both for training and testing data)
				StringToWordVector stw = new StringToWordVector(100000);
				stw.setOutputWordCounts(true);
				stw.setIDFTransform(true);
				stw.setTFTransform(true);
				SnowballStemmer stemmer = new SnowballStemmer();
				stw.setStemmer(stemmer);
				WordsFromFile stopwords = new WordsFromFile();
				stopwords.setStopwords(new File("dic/stopwords.txt"));
				stw.setStopwordsHandler(stopwords);
				Instances trainSet = DataSource.read(trainingDataPath);
				Instances testSet = DataSource.read(testingDataPath);
				stw.setInputFormat(trainSet);
				trainSet = Filter.useFilter(trainSet, stw);
				trainSet.setClassIndex(0);
				testSet = Filter.useFilter(testSet, stw);
				testSet.setClassIndex(0);

				ArffSaver saver = new ArffSaver();
				saver.setInstances(trainSet);
				saver.setFile(new File("./data/" + projects.get(source) + ".arff"));
				saver.writeBatch();

				// attribute selection for training data
				AttributeSelection attSelection = new AttributeSelection();
				Ranker ranker = new Ranker();
				ranker.setNumToSelect((int) (trainSet.numAttributes() * ratio));
				InfoGainAttributeEval ifg = new InfoGainAttributeEval();
				attSelection.setEvaluator(ifg);
				attSelection.setSearch(ranker);
				attSelection.setInputFormat(trainSet);
				trainSet = Filter.useFilter(trainSet, attSelection);
				testSet = Filter.useFilter(testSet, attSelection);

				Classifier classifier = new NaiveBayesMultinomial();

				classifier.buildClassifier(trainSet);

				for (int i = 0; i < testSet.numInstances(); i++) {
					Instance instance = testSet.instance(i);
					double score = 0;
					if (classifier.classifyInstance(instance) == 1.0) {
						score = 1;
					} else
						score = -1;
					eLearner.vote(i, score);

				}

			}

			eLearner.evaluate();

		}

	}

}
