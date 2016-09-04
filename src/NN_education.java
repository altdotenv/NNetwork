import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Random;

import Jama.*;

public class NN_education {
	private final double STEP_SIZE = 0.6;
	private TransferFunction transfer;
	private ArrayList<Neuron> hLayer;
	private ArrayList<Neuron> oLayer;
	private ArrayList<Neuron> iLayer;
	private Matrix weights;
	public int nodeN;
	private final int H_UNIT_NUMBER = 7;
	
	public static void main(String[] args) throws IOException{
		
		NN_education nNetwork = new NN_education(new String[10]);

		nNetwork.backPropagation(inputValues);
		System.out.println("TRAINING COMPLETED! NOW PREDICTING.");
		nNetwork.makePrediction(predictReader, null);
		
	}
	
	private void makePrediction(BufferedReader predictReader, BufferedReader keyReader) throws IOException {
		String line = predictReader.readLine();
		line = predictReader.readLine();
		while (line != null){
			String[] vals = line.split(",");
			Matrix input = new Matrix(1,vals.length);
			for (int i = 0; i < vals.length; i++){
				input.set(0, i, Double.valueOf(vals[i]));
			}
			float result =  (getOutput(input)*100);
			System.out.println(result);
			line = predictReader.readLine();
		}		

	}

	private float getOutput(Matrix input){
		Matrix iLayerValues = new Matrix(1,this.iLayer.size());
		Matrix hLayerValues = new Matrix(1,this.hLayer.size());
		for (int i = 0; i < this.iLayer.size(); i++){
			float output = this.iLayer.get(i).getNeuronValue(input)/100;
			iLayerValues.set(0, i, output);
			this.iLayer.get(i).setOutput(output);
		}
		for (int j = 0; j < this.hLayer.size(); j++){
			Neuron current = this.hLayer.get(j);
			Matrix w = this.weights.getMatrix(oLayer.size() + hLayer.size(),this.weights.getColumnDimension()-1,current.getNodeNumber(),current.getNodeNumber());	
			float output = current.getNeuronValue(iLayerValues.times(w));
			hLayerValues.set(0, j, output);
			current.setOutput(output);
		}

		Matrix w = this.weights.getMatrix(hLayer.get(0).getNodeNumber(),hLayer.get(hLayer.size()-1).getNodeNumber(),0,0);
		return this.oLayer.get(0).getNeuronValue(hLayerValues.times(w));
	}
	

	private void backPropagation(Matrix data) {
		float error= 0;
		float oldError = 10;
		while (Math.abs(oldError - error) > .0000001){
			int i = 0;
			oldError = error;
			error = 0;
			while (i < data.getRowDimension()){
				Matrix input = data.getMatrix(i, i, 0, data.getColumnDimension()-2); //leave out the label
				double result = getOutput(input);
				double label = data.get(i, data.getColumnDimension()-1)/100;
				
				for (Neuron o : this.oLayer) o.setDelta(result*(1-result)*(label-result));
				for (Neuron h : this.hLayer){
					double effect = 0;
					for (Neuron o : this.oLayer){
						effect = effect + o.getDelta()*this.weights.get(h.getNodeNumber(), o.getNodeNumber());
					}
					h.setDelta(effect * h.getOutput() * (1-h.getOutput())); 
				}
				
				for (Neuron o : this.oLayer){
					for (Neuron h : this.hLayer){
						double oldWeight = this.weights.get(h.getNodeNumber(), o.getNodeNumber());
						this.weights.set(h.getNodeNumber(), o.getNodeNumber(), oldWeight + (o.getDelta() * this.STEP_SIZE * h.getOutput()));
						this.weights.set(o.getNodeNumber(), h.getNodeNumber(),  oldWeight + o.getDelta() * this.STEP_SIZE * h.getOutput());
					}
				}
				
				for (Neuron in : this.iLayer){
					for (Neuron h : this.hLayer){
						double oldWeight = this.weights.get(in.getNodeNumber(), h.getNodeNumber());
						this.weights.set(in.getNodeNumber(), h.getNodeNumber(), oldWeight +h.getDelta()*this.STEP_SIZE*in.getOutput());
						this.weights.set(h.getNodeNumber(), in.getNodeNumber(), oldWeight +h.getDelta()*this.STEP_SIZE*in.getOutput());
					}
				}
				
				error += Math.pow(result - label, 2);

				i++;
			}
			System.out.println(error);
		}
	}


	public NN_education(String[] strings){
		this.nodeN = 0;
		this.transfer = new Sigmoid();
		this.oLayer = initializeOutputLayer();
		this.hLayer = initializeHiddenLayer();
		this.iLayer = new ArrayList<Neuron>();
		
		for (int i = 0; i < strings.length-1; i++){
			this.iLayer.add(new InputNeuron(i,this.nodeN,this.transfer));
			this.nodeN++;
		}
		this.iLayer.add(new InputNeuron(strings.length-1,this.nodeN,this.transfer));
		this.nodeN++;
		setWeights(this.nodeN);
	}

//	correctly sets weight
	private void setWeights(int nodeN2) {
		Random rn = new Random(15);
		Matrix weights = new Matrix(nodeN2,nodeN2);
		for (int i = 0; i < nodeN2; i++){
			for (int j = 0; j < nodeN2; j++){
				double initW = rn.nextDouble();
				weights.set(i, j, initW-.5);
				weights.set(j, i, initW-.5);
			}
		}
		this.weights = weights;
	}

	private ArrayList<Neuron> initializeHiddenLayer() {
		ArrayList<Neuron> hiddenArray = new ArrayList<Neuron>();
		for (int i = 0; i < H_UNIT_NUMBER  ; i++){
			hiddenArray.add(new HiddenNeuron(this.nodeN, this.transfer));
			this.nodeN++;
		}
		return hiddenArray; 
	}

//	correctly initialize Output Layer
	private ArrayList<Neuron> initializeOutputLayer() {
		OutputNeuron output = new OutputNeuron(this.nodeN,this.transfer);
		this.nodeN ++;
		ArrayList<Neuron> outputArray = new ArrayList<Neuron>();
		outputArray.add(output);
		return outputArray;
	}
	
}
