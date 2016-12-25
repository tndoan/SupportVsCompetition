package utils;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import main.Model;
import object.UserObject;
import object.VenueObject;

/**
 * calculate the similarity between 2 venues
 * @author tndoan
 *
 */
public class Similarity {
	/**
	 * instance is the cache of similarity one time
	 */
	private static Similarity instance;

	private HashMap<String, Double> cksMap;

	private HashMap<String, Double> distMap;

	private Similarity() {
		cksMap = new HashMap<>();
		distMap = new HashMap<>();
	}

	public static void initialize() {
		instance = new Similarity();
	}
	
	/**
	 * each venue has a vector, say v1, whose size is equal to number of users in dataset.
	 * each element of v1 is equal to the number of check-in between users and venues
	 * the order of users in all vector v is similar.
	 * 
	 * This function will calculate the cosine similarity given 2 ids of venues
	 * @param s1	venue id of 1st venue
	 * @param s2	venue id of 2nd venue
	 * @param m 	model which contains all information of users and venues
	 * @return 		cosine similarity
	 */
	public static double cosinCheckinScore(String s1, String s2, Model m){
		String key = makeKey(s1, s2);
		Double value = instance.cksMap.get(key);
		if (value != null) // check from cache if the cks cosine has been calculated before
			return value;

		VenueObject v1 = m.getVenueObj(s1);
		VenueObject v2 = m.getVenueObj(s2);
		
		double num = 0.0; // numerator
		double d1 = 0.0; // denominator
		double d2 = 0.0;
		
		Set<String> uIds = new HashSet<>();
		uIds.addAll(v1.getUserIds());
		uIds.addAll(v2.getUserIds());
		
		for (String uId : uIds) {
			UserObject uObj = m.getUserObj(uId);
			double n1 = (double) uObj.retrieveNumCks(s1);
			double n2 = (double) uObj.retrieveNumCks(s2);
			
			num += n1 * n2;
			d1 += n1 * n1;
			d2 += n2 * n2;
		}

		double result = num / Math.sqrt(d1 * d2);
		instance.cksMap.put(key, result); // update cache
		return result;
	}
	
	/**
	 * consider 2 venues
	 * each venue has a vector, say v1, whose size is equal to number of users making check-in to both venues.
	 * each element of v1 is equal to the distance between users and venues
	 * the order of users in both vectors v are similar.
	 * 
	 * This function will calculate the cosin similarity given 2 ids of venues
	 * @param s1	venue id of 1st venue
	 * @param s2	venue id of 2nd venue
	 * @param m		model which contains all information of users and venues
	 * @return 		cosine similarity
	 */
	public static double cosinDistanceScore(String s1, String s2, Model m) {
		String key = makeKey(s1, s2);
		Double value = instance.distMap.get(key);
		if (value != null)
			return value;

		double num = 0.0;
		double d1 = 0.0;
		double d2 = 0.0;
		
		VenueObject v1 = m.getVenueObj(s1);
		VenueObject v2 = m.getVenueObj(s2);
		
		Set<String> uIds = new HashSet<>();
		uIds.addAll(v1.getUserIds());
		Set<String> tempU = new HashSet<>();
		tempU.addAll(v2.getUserIds());
		uIds.retainAll(tempU); // uIds is the intersection. It contains the users who check-in to both s1 ans s2
		
		if (uIds.size() == 0)
			return 0.0;
		
		for (String uId : uIds) {
			UserObject uObj = m.getUserObj(uId);
			// plus 0.1 meter because we want to avoid the case of venues which are home location and only check-ined by their owners
			double dis1 = Distance.calculateDistance(uObj.getLocation(), v1.getLocation()) + 0.1;
			double dis2 = Distance.calculateDistance(uObj.getLocation(), v2.getLocation()) + 0.1;
			
			d1 += dis1 * dis1;
			d2 += dis2 * dis2;
			num += dis1 * dis2;
		}
		
		double result = num / Math.sqrt(d1 * d2);
		instance.distMap.put(key, result); // update the cache
		return result;
	}
	
	/**
	 * it returns the cosin similarity between 2 vectors.
	 * @param v1
	 * @param v2 
	 * @return
	 */
	public static double cosinVector(double[] v1, double[] v2) {
		double n = Function.innerProduct(v1, v2);
		double m = Math.sqrt(Function.innerProduct(v1, v1) * Function.innerProduct(v2, v2));
		return n / m;
	}

	/**
	 * From 2 string make a unique key for the map
	 * @param id1	1st string
	 * @param id2	2nd string
	 * @return		a unique key from 2 strings
	 */
	private static String makeKey(String id1, String id2) {
		int rComp = id1.compareTo(id2);

		if (rComp == 1){
			String result = new String(id1);
			return result.concat(id2);
		}

		String result = new String(id2);
		return result.concat(id1);

	}
}
