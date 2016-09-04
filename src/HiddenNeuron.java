import Jama.Matrix;

public class HiddenNeuron implements Neuron{
	
	private int number; 
	private TransferFunction transfer;
	public double delta;
	public double outputV;


	public HiddenNeuron(int number, TransferFunction transfer){
		this.number = number;
		this.transfer = transfer;
	}
	
	@Override
	public int getNodeNumber(){
		return this.number;
	}

	@Override
	public float getNeuronValue(Matrix input) {
		float sum = 0;
		for (int i = 0; i < input.getColumnDimension(); i++){
			for (int j = 0; j < input.getRowDimension(); j++){
				sum += input.get(j,i);
			}
		}
		return transfer.returnValue(sum);
	}

	@Override
	public double getDelta() {
		return this.delta;
	}

	@Override
	public void setDelta(double d) {
		this.delta = (float) d;
	}

	@Override
	public double getOutput() {
		return this.outputV;
	}

	@Override
	public void setOutput(double d) {
		this.outputV = (float)d;
	}

	

}
