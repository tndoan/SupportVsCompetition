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
	 * each venue has a vector, say v1, whose size is equal to number of users in dataset.
	 * each element of v1 is equal to the number of check-in between users and venues
	 * the order of users in all vector v is similar.
	 * 
	 * This function will calculate the cosin similarity given 2 ids of venues
	 * @param s1	venue id of 1st venue
	 * @param s2	venue id of 2nd venue
	 * @param vMap	hash map whose key is venue id, value is venue object
	 * @param uMap	hash map whose key is user id, value is user object
	 * @return 		cosin similarity
	 */
	public static double cosinCheckinsScore(String s1, String s2, HashMap<String, VenueObject> vMap, HashMap<String, UserObject> uMap) {
		VenueObject v1 = vMap.get(s1);
		VenueObject v2 = vMap.get(s2);
		
		double num = 0.0; // numerator
		double d1 = 0.0; // denominator
		double d2 = 0.0;
		
		Set<String> uIds = new HashSet<>();
		uIds.addAll(v1.getUserIds());
		uIds.addAll(v2.getUserIds());
		
		for (String uId : uIds) {
			UserObject uObj = uMap.get(uId);
			double n1 = (double) uObj.retrieveNumCks(s1);
			double n2 = (double) uObj.retrieveNumCks(s2);
			
			num += n1 * n2;
			d1 += n1 * n1;
			d2 += n2 * n2;
		}
		return num / Math.sqrt(d1 * d2);
	}
	
	/**
	 * 
	 * @param s1
	 * @param s2
	 * @param m
	 * @return
	 */
	public static double cosinCheckinScore(String s1, String s2, Model m){
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
		return num / Math.sqrt(d1 * d2);
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
	 * @param vMap	hash map whose key is venue id, value is venue object
	 * @param uMap	hash map whose key is user id, value is user object
	 * @return 		cosin similarity
	 */
	public static double cosinDistanceScore(String s1, String s2, HashMap<String, VenueObject> vMap, HashMap<String, UserObject> uMap) {
		double num = 0.0;
		double d1 = 0.0;
		double d2 = 0.0;
		
		VenueObject v1 = vMap.get(s1);
		VenueObject v2 = vMap.get(s2);
		
		Set<String> uIds = new HashSet<>();
		uIds.addAll(v1.getUserIds());
		Set<String> tempU = new HashSet<>();
		tempU.addAll(v2.getUserIds());
		uIds.retainAll(tempU); // uIds is the intersection. It contains the users who check-in to both s1 ans s2
		
		if (uIds.size() == 0)
			return 0.0;
		
		for (String uId : uIds) {
			UserObject uObj = uMap.get(uId);
			// plus 0.1 meter because we want to avoid the case of venues which are home location and only check-ined by their owners
			double dis1 = Distance.calculateDistance(uObj.getLocation(), v1.getLocation()) + 0.1;
			double dis2 = Distance.calculateDistance(uObj.getLocation(), v2.getLocation()) + 0.1;
			
			d1 += dis1 * dis1;
			d2 += dis2 * dis2;
			num += dis1 * dis2;
		}
		
		if (Double.isNaN(num / Math.sqrt(d1 * d2)))
			System.out.println(num + ";" + d1 + ";" + d2);
		
		return num / Math.sqrt(d1 * d2);
	}
	
	/**
	 * 
	 * @param s1
	 * @param s2
	 * @param m
	 * @return
	 */
	public static double cosinDistanceScore(String s1, String s2, Model m) {
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
		
		return num / Math.sqrt(d1 * d2);
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
}
