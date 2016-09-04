import Jama.Matrix;

public interface Neuron {
	
	public int getNodeNumber();
	
	public float getNeuronValue(Matrix iLayerValues);
	
	public double getDelta();
	
	public void setDelta(double d);
	
	public double getOutput();
	
	public void setOutput(double d);
}
