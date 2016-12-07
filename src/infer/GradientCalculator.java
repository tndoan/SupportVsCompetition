package infer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.IntStream;

import main.ModeSimilarity;
import main.Model;
import main.Parameters;
import object.UserObject;
import object.VenueObject;
import utils.Function;
import utils.Similarity;

public class GradientCalculator {

	private Model m;
	
	private Parameters p;
	
	private int k;
	
	private boolean isFriend;

	public GradientCalculator(Model model, Parameters params) {
		this.m = model;
		this.p = params;
		this.k = m.getK();
		this.isFriend = m.isFriend();
	}

	/**
	 * 
	 * @param uId
	 * @param diff	
	 * @return
	 */
	public double userBias(String uId, double diff) {
		UserObject u = m.getUserObj(uId);
		return diff + u.getBias() * p.getLambda_2();
	}
	
	/**
	 * 
	 * @param vId
	 * @param diff
	 * @return
	 */
	public double venueBias(String vId, double diff) {
		VenueObject v = m.getVenueObj(vId);
		return diff + v.getBias() * p.getLambda_2();
	}
	
	/**
	 * Intrinsic characteristic of venue
	 * @param vId
	 * @param diff
	 * @return
	 */
	public double[] iVenueGrad(String vId) {
		VenueObject v = m.getVenueObj(vId);
		double[] result = new double[k];
		System.arraycopy(v.getIFactors(), 0, result, 0, k);
		IntStream.range(0, k).parallel().forEach(i -> result[i] *= p.getLambda_1());
		
		ArrayList<String> users = v.getUserIds(); // list of users who make check-ins to this venue
		for (String uId : users) {
			UserObject u = m.getUserObj(uId);
			double diff = m.getPredNumCks(uId, vId) - u.retrieveNumCks(vId);
			double[] uVector = Arrays.copyOf(u.getFactors(), k);
			
			IntStream.range(0, k).parallel().forEach(i -> result[i] += uVector[i] * diff);
			
		}
		
		return result;
	}
	
	public double[] userGrad(String uId) {
		double[] result = new double[k];
		UserObject u = m.getUserObj(uId);
		
		double[] uFactors = u.getFactors();
		IntStream.range(0, k).parallel().forEach(i -> result[i] = p.getLambda_1() * uFactors[i]); // regularization
		
		Set<String> venueIds = u.getAllVenues();
		for (String venueId : venueIds) {
			VenueObject v = m.getVenueObj(venueId);
			double diff = m.getPredNumCks(uId, venueId) - u.retrieveNumCks(venueId);
			
			IntStream.range(0, k).parallel().forEach(i -> result[i] += diff * supUserGrad(u, v, i));
//			for (int i = 0; i < k; i++)
//				result[i] += diff * supUserGrad(u, v, i);
			
		}
		
		// TODO: friendship
		ArrayList<String> lOfFriends = u.getListOfFriends();
		if (isFriend && lOfFriends != null) {

			double[] diffFriend = new double[k];
			for (int i = 0; i < k; i++)
				diffFriend[i] = 0.0;

			for (String friend : lOfFriends) {
				UserObject fObj = m.getUserObj(friend);
				if (fObj == null)
					continue;
				double[] fFactors = fObj.getFactors();
				IntStream.range(0,  k).parallel().forEach(i -> diffFriend[i] += uFactors[i] - fFactors[i]);
			}
			IntStream.range(0, k).parallel().forEach(i -> result[i] += p.getLambda_f() * diffFriend[i]);
		}
		
		return result;
	}
	
	public double supUserGrad(UserObject u, VenueObject v, int t) {
		double result = 0.0;
		
		double alpha = m.getAlpha();
		boolean isSigmoid = m.isSigmoid();
		int mode = m.getModeSim();
		
		double[] uFactor = u.getFactors();
		ArrayList<String> neighborIds = v.getNeighbors();
		double UiQj = Function.innerProduct(uFactor, v.getEFactors());
		for (String neighborId : neighborIds) {
			VenueObject neighbor = m.getVenueObj(neighborId);
			double UiQk = Function.innerProduct(uFactor, neighbor.getEFactors());
			double comparison = UiQj - UiQk;
			
			double firstPart = alpha * UiQk; // first part
			if (isSigmoid) {
				double e = Math.exp(comparison);
				double temp = - e / ((1.0 + e) * ( 1.0 + e));
				firstPart *= temp * (v.getEFactors()[t] - neighbor.getEFactors()[t]);
			} else 
				firstPart *= Function.normal(comparison) * (v.getEFactors()[t] - neighbor.getEFactors()[t]);
			
			double secondPart = neighbor.getEFactors()[t]; // second part
			
			double sim = 0.0;
			if (mode == ModeSimilarity.COSIN_CKS_SIM) 
				sim = Similarity.cosinCheckinScore(neighborId, v.getId(), m);
			else if (mode == ModeSimilarity.COSIN_DIST_SIM)
				sim = Similarity.cosinDistanceScore(neighborId, v.getId(), m);
			else //if (mode == ModeSimilarity.CONSTANT)
				sim = 1.0;
			
			if (isSigmoid) 
				secondPart *= (alpha * Function.sigmoidFunction(comparison) + (1.0 - alpha) * sim);
			else
				secondPart *= (alpha * Function.cdf(comparison) + (1.0 - alpha) * sim);
			
			result += firstPart + secondPart;
			
			if (Double.isNaN(secondPart))
				System.out.println(neighborId);
		}
		
		
		
		result *= m.getBeta() / ((double) neighborIds.size());
		result += v.getIFactors()[t];
		
		return result;
	}
	
	/**
	 * 
	 * @param vId
	 * @return
	 */
	public double[] eVenueGrad(String vId) {
		VenueObject v = m.getVenueObj(vId);
		
		double[] result = new double[k];
		System.arraycopy(v.getEFactors(), 0, result, 0, k);
		IntStream.range(0, k).forEach(i -> result[i] *= p.getLambda_3());
		
		ArrayList<String> users = v.getUserIds();
		for (String uId : users) {
			UserObject u = m.getUserObj(uId);
			double diff = m.getPredNumCks(uId, vId) - u.retrieveNumCks(vId);
			
			IntStream.range(0, k).parallel().forEach(i -> result[i] += diff * supEVenueGrad(v, u, i));
		}
		
		return result;
	}
	
	
	private double supEVenueGrad(VenueObject v, UserObject u, int t) {
		int mode = m.getModeSim(); // similarity mode
		boolean isSigmoid = m.isSigmoid();
		double result = 0.0;
		double alpha = m.getAlpha();
		double beta = m.getBeta();
		
		ArrayList<String> neighborIds = v.getNeighbors();
		double[] uFactors = u.getFactors();
		double UiQj = Function.innerProduct(uFactors, v.getEFactors());
		for (String neighborId : neighborIds) {
			VenueObject neighbor = m.getVenueObj(neighborId);
			double UiQk = Function.innerProduct(uFactors, neighbor.getEFactors());
			double comparison = UiQj - UiQk;
			
			double firstPart = alpha * UiQk;
			if (isSigmoid) {
				double e = Math.exp(comparison);
				double temp = e / ((1.0 + e) * (1.0 + e));
				firstPart *= temp; 
			} else {
				firstPart *= (- Function.normal(comparison));
			}
			
			double secondPart = uFactors[t];
			double sim = 0.0;
			if (mode == ModeSimilarity.COSIN_CKS_SIM) {
				sim = Similarity.cosinCheckinScore(neighborId, v.getId(), m);
			} else if (mode == ModeSimilarity.COSIN_DIST_SIM) {
				sim = Similarity.cosinDistanceScore(neighborId, v.getId(), m);
			} else // mode == ModeSimilarity.CONSTANT
				sim = 1.0;
			
			if (isSigmoid) {
				secondPart *= (alpha * Function.sigmoidFunction(comparison) + (1.0 - alpha) * sim); 
			} else { // cdf function
				secondPart *= (alpha * Function.cdf(comparison) + (1.0 - alpha) * sim);
			}
			
			result += firstPart + secondPart;
		}
		
		result *= beta / ((double) neighborIds.size());
		
		return result;
	}
}
