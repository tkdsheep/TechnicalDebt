package model;

import weka.core.Instance;
import weka.core.Instances;

public class EnsembleLearner {
	
	private double tp,fp,fn;
	private double precision,recall,fmeasure;
	private double[] vote;
	private Instances testData;
	
	public EnsembleLearner(Instances testData){
		this.testData = testData;
		vote = new double[testData.numInstances()];
	}
	
	public EnsembleLearner(){
		testData = null;
	}
	
	public void vote(int index, double score){
		vote[index]+=score;
	}
	
	public void evaluate(){
		
		int[] sum = new int [10];
		
		tp = fp = fn = 0;
		for(int i=0;i<testData.numInstances();i++){	
			Instance instance = testData.instance(i);
			double label;
			if(vote[i]>0){	
				label = 1.0;	
				sum[(int)(vote[i])]++;
			}
			else{
				label = 0.0;	
				sum[(int)(-vote[i])]++;
			}
		
			if(instance.classValue()==1.0&&label==1.0)	
				tp++;		
			if(instance.classValue()==1.0&&label==0.0){
				fn++;
				//System.out.println(vote[i]+", "+instance.stringValue(0));
			}
			if(instance.classValue()==0.0&&label==1.0)
				fp++;	
		}
		precision = tp/(tp+fp);
		recall = tp/(tp+fn);
		fmeasure = 2*precision*recall/(precision+recall);
		System.out.println("tp,fp,fn: "+tp+" "+fp+" "+fn);
		System.out.println("precision: "+precision+", recall: "+recall+", f-measure: "+fmeasure);
		
		
	}
	


	public double getTp() {
		return tp;
	}

	public void setTp(double tp) {
		this.tp = tp;
	}

	public double getFp() {
		return fp;
	}

	public void setFp(double fp) {
		this.fp = fp;
	}

	public double getFn() {
		return fn;
	}

	public void setFn(double fn) {
		this.fn = fn;
	}

	public double getPrecision() {
		return precision;
	}

	public void setPrecision(double precision) {
		this.precision = precision;
	}

	public double getRecall() {
		return recall;
	}

	public void setRecall(double recall) {
		this.recall = recall;
	}

	public double getFmeasure() {
		return fmeasure;
	}

	public void setFmeasure(double fmeasure) {
		this.fmeasure = fmeasure;
	}

	public double[] getVote() {
		return vote;
	}

	public void setVote(double[] vote) {
		this.vote = vote;
	}

	public Instances getTestData() {
		return testData;
	}

	public void setTestData(Instances testData) {
		this.testData = testData;
	}
	
	
	

}
