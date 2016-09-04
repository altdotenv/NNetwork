
public class Sigmoid implements TransferFunction{

	@Override
	public float returnValue(float input) {
//		System.out.println(input);
//		System.out.println( (1.0/(1+Math.exp(-input))));
//		System.out.println("SIgmoid done");
//		
		return (float) (1.0/(1+Math.exp(-input)));
	}

}
