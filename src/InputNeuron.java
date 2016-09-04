import Jama.Matrix;

public class InputNeuron implements Neuron{
	private int label;
	private int number; 
	private TransferFunction transfer;
	private float delta;
	private float outputV;
	
	public InputNeuron(int label, int number, TransferFunction transfer){
		this.label = label;
		this.number = number;
		this.transfer = transfer;
	}
	
	@Override
	public int getNodeNumber(){
		return this.number;
	}

	@Override
	public float getNeuronValue(Matrix input) {
		if (this.label == 0) return (float) (input.get(0, this.label)-1900)/100;
		if (this.label == 1) return (float) (input.get(0, this.label))/10;
		if (this.label == input.getColumnDimension()) return 1;
		return (float) input.get(0,this.label);
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
