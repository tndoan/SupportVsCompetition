package object;

import java.util.ArrayList;
import java.util.Random;

/**
 * 
 * @author tndoan
 *
 */
public class VenueObject {
	/**
	 * 
	 * @param id
	 * @param totalCks
	 * @param location
	 * @param neighbors
	 * @param userIds
	 * @param k				number of latent factors for intrinsic and extrinsic vectors
	 */
	public VenueObject(String id, int totalCks,	PointObject location, ArrayList<String> neighbors, 
			ArrayList<String> userIds, int k){
		Random r = new Random();
		this.id = id;
		this.location = location;
		this.neighbors = neighbors;
		this.userIds = userIds;
		this.totalCks = totalCks;
		
		this.iFactors = new double[k];
		for (int i = 0; i < k; i++)
			iFactors[i] = r.nextDouble() * 0.1;
		
		this.eFactors = new double[k];
		for (int i = 0; i < k; i++)
			eFactors[i] = r.nextDouble() * 0.1;
		
		this.bias = 0.0;
	}
	
	/**
	 * extrinsic characteristic of venue
	 */
	private double[] eFactors;
	
	/**
	 * return extrinsic characteristic of venue
	 * @return
	 */
	public double[] getEFactors() {
		return eFactors;
	}

	/**
	 * set extrinsic characteristic of venue
	 * @param eFactors
	 */
	public void setEFactors(double[] eFactors) {
		this.eFactors = eFactors;
	}

	/**
	 * intrinsic characteristic of venue
	 */
	private double[] iFactors;
	
	/**
	 * get the intrinsic latent vector of venue
	 * @return
	 */
	public double[] getIFactors() {
		return iFactors;
	}

	/**
	 * set the intrinsic latent vector of venue
	 * @param factors
	 */
	public void setIFactors(double[] factors) {
		this.iFactors = factors;
	}

	/**
	 * total number of check-in that it has
	 */
	private int totalCks;
	
	/**
	 * location of venue
	 */
	private PointObject location;
	
	/**
	 * id of venue
	 */
	private String id;
	
	/**
	 * list of id of neighbors
	 */
	private ArrayList<String> neighbors;
	
	/**
	 * list of user ids who have check-in in this venue
	 */
	private ArrayList<String> userIds;

	public PointObject getLocation() {
		return location;
	}

	public String getId() {
		return id;
	}

	public ArrayList<String> getNeighbors() {
		return neighbors;
	}

	public ArrayList<String> getUserIds() {
		return userIds;
	}

	public int getTotalCks() {
		return totalCks;
	}
	
	private double bias;

	/**
	 * get the bias of venue
	 * @return
	 */
	public double getBias() {
		return bias;
	}

	/**
	 * set the bias of venue
	 * @param bias
	 */
	public void setBias(double bias) {
		this.bias = bias;
	}
	
	
	
}
