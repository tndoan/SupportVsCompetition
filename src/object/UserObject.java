package object;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;

public class UserObject {
	/**
	 * home location of user
	 */
	private PointObject location;
	
	/**
	 * checkin map whose key is venue id, value is number of check-in that user has made in this venue
	 */
	private HashMap<String, Double> checkinMap;
	
	/**
	 * id of user
	 */
	private String id;
	
	/**
	 * latent factor vector
	 */
	private double[] factors;
	
	/**
	 * bias parameter of user
	 */
	private double bias;
	
	private ArrayList<String> listOfFriends;
	
	/**
	 * 
	 * @return	list of id of his friends
	 */
	public ArrayList<String> getListOfFriends() {
		return listOfFriends;
	}

	/**
	 * get the value of bias parameter
	 * @return
	 */
	public double getBias() {
		return bias;
	}

	/**
	 * function to set the bias parameter
	 * @param bias
	 */
	public void setBias(double bias) {
		this.bias = bias;
	}

	public double[] getFactors() {
		return factors;
	}

	public void setFactors(double[] factors) {
		this.factors = factors;
	}

	/**
	 * get how many check-in user has done in this venue
	 * @param vIds	venue id
	 * @return		number of check-in
	 */
	public double retrieveNumCks(String vIds){
		Double num = checkinMap.get(vIds);
		if (num == null)
			return 0.0;
		else
			return num;
	}

	public PointObject getLocation() {
		return location;
	}

	public String getId() {
		return id;
	}
	
	/**
	 * 
	 * @param id
	 * @param location
	 * @param checkinMap
	 * @param k				number of latent factors
	 */
	public UserObject(String id, PointObject location, HashMap<String, Double> checkinMap, int k, ArrayList<String> listOfFriend){
		Random r = new Random();
		this.id = id;
		this.location = location;
		this.checkinMap = checkinMap;
		
		this.factors = new double[k];
		for (int i = 0; i < k; i++)
			factors[i] = r.nextDouble() * 0.1;
		
		this.bias = 0.0;
		this.listOfFriends = listOfFriend;
	}
	
	/**
	 * 
	 * @return the set of venue id where user has done check-in
	 */
	public Set<String> getAllVenues() {
		return checkinMap.keySet();
	}
}
