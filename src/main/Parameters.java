package main;

public class Parameters {
	/**
	 * regularization for users and venues
	 */
	private double lambda_1;
	
	/**
	 * regularization of bias term of users and venues
	 */
	private double lambda_2;
	
	/**
	 * regularization of extrinsic characteristic vector of venue
	 */
	private double lambda_3;
	
	/**
	 * regularization of friends
	 */
	private double lambda_f;
	
	/**
	 * create parameters
	 * @param lambda_1	regularization for users and venues
	 * @param lambda_2	regularization of bias term of users and venues
	 * @param lambda_f	regularization of friends
	 */
	public Parameters(double lambda_1, double lambda_2, double lambda_3, double lambda_f) {
		this.lambda_1 = lambda_1;
		this.lambda_2 = lambda_2;
		this.lambda_3 = lambda_3;
		this.lambda_f = lambda_f;
	}

	/**
	 * 
	 * @return	regularization for users and venues
	 */
	public double getLambda_1() {
		return lambda_1;
	}

	/**
	 * 
	 * @return	regularization of bias term of users and venues
	 */
	public double getLambda_2() {
		return lambda_2;
	}

	/**
	 * 
	 * @return	regularization of friends
	 */
	public double getLambda_f() {
		return lambda_f;
	}

	/**
	 * 
	 * @return	regularization of extrinsic characteristic vector of venue
	 */
	public double getLambda_3() {
		return lambda_3;
	}

}
