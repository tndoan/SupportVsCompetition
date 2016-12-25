package main;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import object.PointObject;
import object.UserObject;
import object.VenueObject;
import utils.ReadFile;
import utils.Utils;

public class Prediction extends Model{

	private HashMap<String, HashMap<String, Double>> gt;
	
	public Prediction(String outputFName, String uFile, String vFile, String nFile, String fFile, String cksFile, String groundTruthFName) 
			throws FileNotFoundException, IOException{
		
		// initialize
		venueMap = new HashMap<>();
		userMap = new HashMap<>();
		cksMap = ReadFile.readNumCksFile(cksFile);
		HashMap<String, ArrayList<String>> friendInfo = new HashMap<>();

		// read data from files
		HashMap<String, String> vInfo = ReadFile.readLocation(vFile);
		HashMap<String, String> uInfo = new HashMap<>();
		if (uFile != null)
			uInfo = ReadFile.readLocation(uFile);
		HashMap<String, ArrayList<String>> neighborsInfo = ReadFile.readNeighbors(nFile);
		HashMap<String, ArrayList<String>> userOfVenueMap = Utils.collectUsers(cksMap);
		
		try(BufferedReader br = new BufferedReader(new FileReader(outputFName))) {
		    String line = br.readLine(); // meta info of model
		    parseFirstLine(line);
		    
		    if (isFriend)
		    	friendInfo = ReadFile.readFriendship(fFile);

		    line = br.readLine(); // "users:"
		    line = br.readLine();
		    while (!line.equals("venues:")) {
		        String[] comp = line.split(" ");
		        String userId = comp[0];
		        UserObject u = null;
		        
		        PointObject p = null;
		        if (uFile != null)
		        	p = new PointObject(uInfo.get(userId));
		        
		        if (isFriend)
		        	u = parseUser(userId, line.substring(userId.length() + 1), p, friendInfo.get(userId));
		        else 
		        	u = parseUser(userId, line.substring(userId.length() + 1), p, null);
		        
		        userMap.put(userId, u);
		        line = br.readLine();
		    }
		    
//		    System.out.println("# users:" + userMap.keySet().size());
		    
		    line = br.readLine();
		    while (line != null){
		    	String[] comp = line.split(" ");
		    	String venueId = comp[0];
		    	VenueObject v = parseVenue(venueId, line.substring(venueId.length() + 1), 
		    			new PointObject(vInfo.get(venueId)), neighborsInfo.get(venueId),
		    			userOfVenueMap.get(venueId));
		    	venueMap.put(venueId, v);
		    	line = br.readLine();
		    }
//		    System.out.println("# venues:" + venueMap.keySet().size());
		}
		
		gt = readGroundTruth(groundTruthFName);
	}
	
	private VenueObject parseVenue(String venueId, String info, PointObject location, ArrayList<String> neighbor, ArrayList<String> users) {
		String[] comp = info.split(",\\[");
		double bias = Double.parseDouble(comp[0]);
		double[] efactors = Utils.fromString("[" + comp[1]);
		double[] ifactors = Utils.fromString("[" + comp[2]);
		assert(efactors.length == k);
		assert(ifactors.length == k);
		
		int totalCks = 0;
		for (String u : users)
			totalCks += (int) userMap.get(u).retrieveNumCks(venueId);
		
		VenueObject v = new VenueObject(venueId, totalCks, location, neighbor, users, k);
		v.setBias(bias);
		v.setEFactors(efactors);
		v.setIFactors(ifactors);
		return v;
	}
	
	private UserObject parseUser(String userId, String info, PointObject location, ArrayList<String> friends) {
		String[] comp = info.split(",\\[");
		double bias = Double.parseDouble(comp[0]);
		double[] factors = Utils.fromString("[" + comp[1]);
		assert(factors.length == k); // ensure the length of factor vector is equal to k 
		
		UserObject u =  new UserObject(userId, location, cksMap.get(userId), k, friends);
		u.setBias(bias);
		u.setFactors(factors);
		return u;
	}
	
	private HashMap<String, HashMap<String, Double>> readGroundTruth(String fname) throws IOException {
		HashMap<String, HashMap<String, Double>> result = new HashMap<>();
		
		try(BufferedReader br = new BufferedReader(new FileReader(fname))) {
		    String line = br.readLine(); // meta info of model
 
		    while (line != null) {
		        String[] comp = line.split(",");
		        
		        double numCks = Double.parseDouble(comp[0]);
		        String userId = comp[1];
		        String vId = comp[2];
		        
		        HashMap<String, Double> v = result.get(userId);
		        if (v == null) {
		        	v = new HashMap<>();
		        	result.put(userId, v);
		        }
		        v.put(vId, numCks);
		        
		        line = br.readLine();
		    }
		}
		return result;
	}
	
	private void parseFirstLine(String line) {		
		String[] comp = line.split(";");
		
		// k
		String[] kInfo = comp[0].split("=");
		k = Integer.parseInt(kInfo[1]);
		
		// isSigmoid
		String[] sInfo = comp[1].split("=");
		isSigmoid = Boolean.parseBoolean(sInfo[1]);
		
		// modeSim
		String[] mSim = comp[2].split("=");
		modeSim = Integer.parseInt(mSim[1]);
		
		// beta
		String[] b = comp[3].split("=");
		beta = Double.parseDouble(b[1]);
		
		// alpha 
		String[] a = comp[4].split("=");
		alpha = Double.parseDouble(a[1]);
		
		// isFriend
		String[] fInfo = comp[5].split("=");
		isFriend = Boolean.parseBoolean(fInfo[1]);
		
		// lambda
		double lambda1 = Double.parseDouble(comp[6].split("=")[1]);
		double lambda2 = Double.parseDouble(comp[7].split("=")[1]);
		double lambda3 = Double.parseDouble(comp[8].split("=")[1]);
		double lambdaf = Double.parseDouble(comp[9].split("=")[1]);
		params = new Parameters(lambda1, lambda2, lambda3, lambdaf);
		
		// mu
		mu = Double.parseDouble(comp[10].split("=")[1]);
	}
	
	public void metricOfSIGIR2014() { // see the paper sigir 2014 
		double mae = 0.0;
		double rmse = 0.0;
		double count = 0.0;
		
		for (String userId : gt.keySet()) {
			HashMap<String, Double> g = gt.get(userId);
			for (String venueId : g.keySet()) {
				UserObject u = userMap.get(userId);
				VenueObject v = venueMap.get(venueId);
				if (u == null || v == null)
					continue;
				double pred = calculatePredictedCks(userId, venueId);
				double actual = g.get(venueId);
				count += 1.0;
				
				double diff = pred - actual;
				mae += Math.abs(diff);
				rmse += diff * diff;
			}
		}
		
		System.out.println((mae / count) + "," + (rmse / count));
	}
	
	public void globalMean() {
		double mae = 0.0;
		double rmse = 0.0;
		double count = 0.0;
		
		for (String userId : gt.keySet()) {
			HashMap<String, Double> g = gt.get(userId);
			for (String venueId : g.keySet()) {
				UserObject u = userMap.get(userId);
				VenueObject v = venueMap.get(venueId);
				if (u == null || v == null)
					continue;
				double actual = g.get(venueId);
				count += 1.0;
				
				double diff = mu - actual;
				mae += Math.abs(diff);
				rmse += diff * diff;
			}
		}
		
		System.out.println("mu MAE: " + (mae / count) + " RMSE: " + (rmse / count));
	}
	
	public void resultOfTopK(int[] topk) {
		Arrays.sort(topk);
		int maxTopk = topk[topk.length - 1];
		double count = 0.0;
		
		double[] result = new double[topk.length];
		
		for (String userId : userMap.keySet()) {
			// prediction
			ArrayList<PairObject> list = new ArrayList<>();
			
			for (String venueId : venueMap.keySet()) {
				double pred = calculatePredictedCks(userId, venueId);
				list.add(new PairObject(venueId, pred));
			}
			Set<String> groundTruth = gt.get(userId).keySet();
			ArrayList<String> topkList = topKVenues(list, maxTopk);
			
			// compare to groundtruth
			for (int i = 0; i < topk.length; i++) {
				int tk = topk[i];
				Set<String> t = new HashSet<>(topkList.subList(0, tk));
				Set<String> g = new HashSet<>(groundTruth);
				
				Set<String> intersection = new HashSet<>();
				intersection.addAll(t);
				intersection.retainAll(g);
				
				result[i] += (double) intersection.size() / (double) groundTruth.size();
			}
			
			count += 1.0;
		}
		
		for (int i = 0; i < topk.length; i++) {
			double r = result[i] / count;
			System.out.println("Top " + topk[i] + ":" + r);
		}
	}
	
	public void uiMean() {
		HashMap<String, Double> uMean = new HashMap<>();
		
		for (String uId : cksMap.keySet()) {
			HashMap<String, Double> vM = cksMap.get(uId);
			double count = 0.0;
			double total = 0.0;
			for (String vId : vM.keySet()) {
				total += vM.get(vId);
				count += 1.0;
			}
			uMean.put(uId, total / count);
		}
		
		double mae = 0.0;
		double rmse = 0.0;
		double count = 0.0;
		
		for (String userId : gt.keySet()) {
			HashMap<String, Double> g = gt.get(userId);
			Double pred = uMean.get(userId);
			if (pred == null) continue;
			for (String venueId : g.keySet()) {
				UserObject u = userMap.get(userId);
				VenueObject v = venueMap.get(venueId);
				if (u == null || v == null)
					continue;
				double actual = g.get(venueId);
				count += 1.0;
				
				double diff = pred - actual;
				mae += Math.abs(diff);
				rmse += diff * diff;
			}
		}
		
//		System.out.println("uMean MAE: " + (mae / count) + " RMSE: " + (rmse / count));
		
		mae = 0.0;
		rmse = 0.0;
		count = 0.0;
		for (String userId : gt.keySet()) {
			HashMap<String, Double> g = gt.get(userId);
			for (String venueId : g.keySet()) {
				UserObject u = userMap.get(userId);
				VenueObject v = venueMap.get(venueId);
				if (u == null || v == null)
					continue;
				double pred = (double)v.getTotalCks() / (double)v.getUserIds().size();
				double actual = g.get(venueId);
				count += 1.0;
				
				double diff = pred - actual;
				mae += Math.abs(diff);
				rmse += diff * diff;
			}
		}
		System.out.println("vMean MAE: " + (mae / count) + " RMSE: " + (rmse / count));
	}
	
    private static ArrayList<String> topKVenues(ArrayList<PairObject> orig, int nummax) {
        Collections.sort(orig, new Comparator<PairObject>() {

			@Override
			public int compare(PairObject o1, PairObject o2) {
				if (o1.cks == o2.cks)
					return 0;
		        return o1.cks < o2.cks ? 1 : -1;
			}
		});

        List<PairObject> p = orig;
        if (orig.size() >= nummax)
        	p = orig.subList(0, nummax);
        
        ArrayList<String> result = new ArrayList<>();
        for (int i = 0; i < p.size(); i++) 
        	result.add(p.get(i).venueId);
        return result;
    }
    
    public static void main(String[] args) {
//    	ArrayList<PairObject> orig = new ArrayList<>();
//    	double[] result = new double[10];
//    	Random rand = new Random();
//    	for (int i = 0; i < 10; i++) {
//    		double cks = (double)rand.nextInt(100);
//    		orig.add(new PairObject(
//    				String.valueOf(i), 
//    				cks
//    				));
//    		result[i] = cks;
//    		System.out.println(i + ":" + cks);
//    	}
//    	
//    	System.out.println("====");
//    	System.out.println(Arrays.toString(topKVenues(orig, 3).toArray()));
//    	Arrays.sort(result);
//    	System.out.println(Arrays.toString(result));
    	
        Set<Integer> a = new HashSet<>(Arrays.asList(new Integer[]{0,2,4,5,6,8,10}));
        Set<Integer> b = new HashSet<>(Arrays.asList(new Integer[]{0,20,40,5,6,8,10}));
        
        Set<Integer> c = new HashSet<>();
        c.addAll(a);
        c.addAll(b);
        
        Set<Integer> d = new HashSet<>();
        d.addAll(a);
        d.retainAll(b);
        
        System.out.println(c.size());
        System.out.println(d.size());
    }
}

class PairObject {
	String venueId;
	double cks;
	
	public PairObject(String venueId, double cks) {
		this.venueId = venueId;
		this.cks = cks;
	}
}