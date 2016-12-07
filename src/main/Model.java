package main;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import infer.GradientCalculator;
import object.PointObject;
import object.UserObject;
import object.VenueObject;
import utils.Function;
import utils.ReadFile;
import utils.Similarity;
import utils.Utils;

public class Model {
	
	/**
	 * isSigmoid = true => sigmoid function
	 * isSigmoid = false => CDF function
	 */
	private boolean isSigmoid;
	
	public boolean isSigmoid() {
		return isSigmoid;
	}

	/**
	 * modeSim is in class ModeSimilarity class
	 */
	private int modeSim;
	
	public int getModeSim() {
		return modeSim;
	}

	/**
	 * beta parameter (refer the paper)
	 */
	private double beta;
	
	/**
	 * get value of beta parameter
	 * @return
	 */
	public double getBeta() {
		return beta;
	}

	/**
	 * alpha parameter (refer the paper)
	 */
	private double alpha;
	
	/**
	 * get value of alpha parameter
	 * @return
	 */
	public double getAlpha() {
		return alpha;
	}

	/**
	 * number of latent factors
	 */
	private int k;
	
	public int getK() {
		return k;
	}
	
	/**
	 * contains all regularization parameters
	 */
	private Parameters params;
	
	/**
	 * indicate if we want to use friendship to model objective function 
	 */
	private boolean isFriend;
	
	public boolean isFriend() {
		return isFriend;
	}

	/**
	 * average number of check-ins in the training data
	 */
	private double mu;
	
	/**
	 * key is venue id, value is venue object corresponding to the venue id
	 */
	private HashMap<String, VenueObject> venueMap;
	
	/**
	 * key is user id, value is user object corresponding to user id
	 */
	private HashMap<String, UserObject> userMap;
	
	/**
	 * check-in map of training
	 * key is user id, value is the map whose key is venue id, value is the number of check-in between user and venue
	 */
	private HashMap<String, HashMap<String, Double>> cksMap;
	
	/**
	 * similar to cksMap but it stores the prediction between user and venue
	 */
	private HashMap<String, HashMap<String, Double>> predCksMap;

	/**
	 * create model
	 * @param uFile		file name of location of users
	 * @param vFile		file name of location of venues
	 * @param nFile		file name of neighbors of each venue
	 * @param fFile		file name of friendship
	 * @param cksFile	file name of check-ins information
	 * @param isSigmoid	use sigmoid or cdf
	 * @param modeSim	the mode of similarity
	 * @param k			number of latent factors
	 * @param alpha		alpha parameter. It must be between 0 and 1
	 * @param beta		beta parameter. It must be between 0 and 1
	 * @param isFriend	indicate if we want to use friend or not
	 * @param params	object that contains all regularization parameters
	 */
	public Model(String uFile, String vFile, String nFile, String fFile, String cksFile, boolean isSigmoid, int modeSim, int k, 
			double alpha, double beta, boolean isFriend, Parameters params) {
		assert(modeSim == ModeSimilarity.COSIN || modeSim == ModeSimilarity.COSIN_CKS_SIM 
				|| modeSim == ModeSimilarity.COSIN_DIST_SIM || modeSim == ModeSimilarity.CONSTANT);
		assert(alpha >= 0 && alpha <= 1 );
		assert(beta >= 0 && beta <= 1);
		this.modeSim = modeSim;
		this.isSigmoid = isSigmoid;
		this.k = k;
		this.alpha = alpha;
		this.beta = beta;
		this.isFriend = isFriend;
		this.params = params;
		
		// initialize 
		venueMap = new HashMap<>();
		userMap = new HashMap<>();
		
		// read data from files
		HashMap<String, String> vInfo = ReadFile.readLocation(vFile);
		
		HashMap<String, String> uInfo = new HashMap<>(); 
		if (uFile != null)
			uInfo = ReadFile.readLocation(uFile);
		
		HashMap<String, ArrayList<String>> neighborsInfo = ReadFile.readNeighbors(nFile);
		
		HashMap<String, ArrayList<String>> friendInfo = new HashMap<>();
		// TODO: friendship here
		if (isFriend)
			friendInfo = ReadFile.readFriendship(fFile);			
		
		this.cksMap = ReadFile.readNumCksFile(cksFile);
		predCksMap = initialziePred();
		
		HashMap<String, ArrayList<String>> userOfVenueMap = Utils.collectUsers(cksMap);
		
		this.mu = calculateMu(cksMap);
		
		Set<String> allUsers = cksMap.keySet();
		if (uFile != null)
			allUsers = uInfo.keySet();
		
		// making user objects
		for(String userId : allUsers){
			
			// parse the location of users
			PointObject location = null;
			if (uFile != null) {
				String locInfo = uInfo.get(userId); // in this context, we know the location of every users
				location = new PointObject(locInfo);
			}
			
			HashMap<String, Double> checkinMap = cksMap.get(userId);

			UserObject uo = new UserObject(userId, location, checkinMap, k, friendInfo.get(userId));
			userMap.put(userId, uo);
		}
		
		// making venue objects
		for (String venueId : vInfo.keySet()) {
			// location
			String locInfo = vInfo.get(venueId);
			PointObject location = new PointObject(locInfo);
			
			// users who cks in this venue
			ArrayList<String> users = userOfVenueMap.get(venueId);
			
			if (users == null)
				continue; // we dont care this case because this venue has no check-ins
			
			// total number of cks
			int cks = 0;
			for (String user : users) {
				cks += userMap.get(user).retrieveNumCks(venueId);
			}
			
			// neighbors
			ArrayList<String> neighbors = neighborsInfo.get(venueId);
			
			VenueObject vo = new VenueObject(venueId, cks, location, neighbors, users, k);
			venueMap.put(venueId, vo);
		}
	}
	
	/**
	 * initialize the prediction map
	 * @return
	 */
	private HashMap<String, HashMap<String, Double>> initialziePred() {
		HashMap<String, HashMap<String, Double>> result = new HashMap<>();
		for (String userId : cksMap.keySet()) {
			HashMap<String, Double> sub = new HashMap<>();
			HashMap<String, Double> venueM = cksMap.get(userId);
			for (String venueId : venueM.keySet())
				sub.put(venueId, 0.0);
			result.put(userId, sub);
		}
		return result;
	}
	
	/**
	 * get the average check-ins between users and venues
	 * @param cksMap	check-in map
	 * @return			the average number of check-ins
	 */
	private double calculateMu(HashMap<String, HashMap<String, Double>> cksMap) {
		double totalPairs = 0.0;
		double totalCks = 0.0;
		
		for (String userId : cksMap.keySet()) {
			HashMap<String, Double> map = cksMap.get(userId);
			for (String venueId : map.keySet()) {
				totalCks += map.get(venueId);
				totalPairs++;
			}
		}
		
		return totalCks / totalPairs;
	}

	/**
	 * calculate the RMSE of our prediction and actual result
	 * @return	the RMSE
	 */
	private double calculateRMSE() {
		double result = 0.0;
		Set<String> allUserIds = userMap.keySet();
		
		for (String uId : allUserIds) {
			HashMap<String, Double> vM = cksMap.get(uId);
			for (String vId : vM.keySet()) {
				double diff = vM.get(vId) - calculatePredictedCks(uId, vId);
				result += diff * diff;
			}
		}
		
		return result;
	}
	
	private double objectiveFunc() { // we dont multiply to 1/2 because it is not necessary
		double result = calculateRMSE();
		
		// user regularization
		Collection<UserObject> allUsers = userMap.values();
		double uReg = 0.0;
		double rB = 0.0; // bias regularization
		for (UserObject u : allUsers) {
			double[] vector = u.getFactors();
			uReg += Function.innerProduct(vector, vector);
			rB += u.getBias() * u.getBias();
		}
		result += params.getLambda_1() * uReg + params.getLambda_2() * rB;
		
		// venue regularization
		Collection<VenueObject> allVenues = venueMap.values();
		double vReg = 0.0; // regularization for intrinsic characteristic vector 
		double eVReg = 0.0; // regularization for extrinsic characteristic vector
		double rB_j = 0.0; // venue bias regularization
		for (VenueObject v : allVenues) {
			double[] eVector = v.getEFactors();
			double[] iVector = v.getIFactors();
			
			eVReg += Function.innerProduct(eVector, eVector);
			vReg += Function.innerProduct(iVector, iVector);
			rB_j += v.getBias() * v.getBias();
		}
		
		result += params.getLambda_1() * vReg + params.getLambda_3() * eVReg + params.getLambda_2() * rB_j;
		
		if (isFriend) {
			HashSet<String> processedUsers = new HashSet<>();
			double fReg = 0.0;
			for (UserObject u : allUsers) {
				ArrayList<String> friends = u.getListOfFriends();
				if (friends == null)
					continue;
				double[] uF = u.getFactors();
				for (String fId : friends) {
					if (processedUsers.contains(fId)) // the pair (u.getId(), fId) has been processed before
						continue;
					processedUsers.add(fId); // we dont want to process this pair latter
					UserObject friend = userMap.get(fId);
					if (friend == null) // for the case of that user does not exist (for ex: users not in training set)
						continue;
					double[] diff = Function.minus(uF, friend.getFactors());
					fReg += Function.innerProduct(diff, diff);
				}
			}
			
			result += params.getLambda_f() * fReg;
		}
		
		return result;
	}
	
	/**
	 * given user and venue id. Predict the number of check-ins between them using our model
	 * @param uId	user id
	 * @param vId	venue id
	 * @return		predicted number of check-ins between them
	 */
	private double calculatePredictedCks(String uId, String vId) {
		VenueObject v = venueMap.get(vId);
		UserObject u = userMap.get(uId);
		
		double innerProdOfUV = Function.innerProduct(v.getIFactors(), u.getFactors());
		double result = mu + v.getBias() + u.getBias() + innerProdOfUV;
		
		double s = 0.0;
		ArrayList<String> neighbors = v.getNeighbors();
		for (String nId : neighbors) {
			VenueObject neighbor = venueMap.get(nId);
			double similarity = 0.0; double competition = 0.0;
			double innerProdOfU_eV = Function.innerProduct(u.getFactors(), v.getEFactors());
			double innerProdOfUN = Function.innerProduct(u.getFactors(), neighbor.getEFactors());
			
			if (isSigmoid) // competition 
				competition = Function.sigmoidFunction(innerProdOfU_eV - innerProdOfUN);
			else
				competition = Function.cdf(innerProdOfU_eV - innerProdOfUN);
			
			if (modeSim == ModeSimilarity.COSIN) // similarity (spatial homophily)  
				similarity = Similarity.cosinVector(v.getEFactors(), neighbor.getEFactors());
			else if (modeSim == ModeSimilarity.COSIN_CKS_SIM)
				similarity = Similarity.cosinCheckinsScore(vId, nId, venueMap, userMap);
			else if (modeSim == ModeSimilarity.COSIN_DIST_SIM)
				similarity = Similarity.cosinDistanceScore(vId, nId, venueMap, userMap);
			else //if (modeSim == ModeSimilarity.CONSTANT)
				similarity = 1.0;
						
			s += (alpha * competition + (1 - alpha) * similarity) * innerProdOfUN;
		}
		
		double numberOfNeighbors = (double) neighbors.size();
		return result + (beta / numberOfNeighbors) * s;
	}
	
	/**
	 * 
	 * @param uId
	 * @return
	 */
	public UserObject getUserObj(String uId) {
		return userMap.get(uId);
	}
	
	/**
	 * 
	 * @param vId
	 * @return
	 */
	public VenueObject getVenueObj(String vId){
		return venueMap.get(vId);
	}
	
	/**
	 * 
	 * @param uId
	 * @param vId
	 * @return
	 */
	public double getPredNumCks(String uId, String vId) {
		HashMap<String, Double> uM = predCksMap.get(uId);
		if (uM == null)
			return 0.0;
		Double d = uM.get(vId);
		if (d == null)
			return 0.0;
		return d;
	}
	
	// TODO
	public void optimization() {
		double prevObjFunc = 0.0;
		boolean isConv = false;
		
		GradientCalculator gc = new GradientCalculator(this, params);
//		double learningRate = 0.00001;
		double learningRate = 0.000001;
		
		int numIter = 0;
		while(!isConv) {
			// calculate diff 
			double diff = 0.0;
			for (String uId : cksMap.keySet()) {
				HashMap<String, Double> uM = cksMap.get(uId);
				HashMap<String, Double> puM = predCksMap.get(uId);
				for (String vId : uM.keySet()) {
					double numCks = uM.get(vId);
					Double pred = puM.get(vId);
					if (pred == null)
						continue;
					diff += (pred - numCks);
				}
			}
			
			// update user
			for (String userId : userMap.keySet()) {
				UserObject uo = getUserObj(userId);
				
				// update bias
				double bias = uo.getBias();
				double grad = gc.userBias(userId, diff);
				uo.setBias(bias - learningRate * grad);
				
				// update user vector
				double[] vector = uo.getFactors();
				double[] vgrad = gc.userGrad(userId);
				uo.setFactors(Function.minus(vector, Function.multiply(learningRate, vgrad)));
				
			}
			
//			UserObject uo = getUserObj("1380522");
//			double[] f = uo.getFactors();
//			System.out.println("Bias:" + uo.getBias());
//			for (int i = 0; i < k; i++)
//				System.out.println(f[i]);
//			System.out.println("=====");
			
			// update venue
			for (String venueId : venueMap.keySet()) {
				VenueObject vo = getVenueObj(venueId);
				
				// update bias
				double bias = vo.getBias();
				double grad = gc.venueBias(venueId, diff);
				vo.setBias(bias - learningRate * grad);
				
				// update intrinsic characters
				double[] iVector = vo.getIFactors();
				double[] iGrad = gc.iVenueGrad(venueId);
				vo.setIFactors(Function.minus(iVector, Function.multiply(learningRate, iGrad)));
				
				// update extrinsic characters
				double[] eVector = vo.getEFactors();
				double[] eGrad = gc.eVenueGrad(venueId);
				vo.setEFactors(Function.minus(eVector, Function.multiply(learningRate, eGrad)));
			}
			
			// update prediction
			for (String uId : cksMap.keySet()) {
				HashMap<String, Double> uM = cksMap.get(uId);
				HashMap<String, Double> puM = predCksMap.get(uId);
				
				if (puM == null) {
					puM = new HashMap<>();
					predCksMap.put(uId, puM);
				}
				
				for (String vId : uM.keySet()) {
					double pred = calculatePredictedCks(uId, vId);
					puM.put(vId, pred);
				}
			}
			
			// check convergence
			double curObjFunc = objectiveFunc();
			System.out.println("Objective function: " + curObjFunc);
//			if (Math.abs(curObjFunc - prevObjFunc) <  0.1 * prevObjFunc)
			if (numIter == 20)
				isConv = true;
			prevObjFunc = curObjFunc;
			numIter++;
		}
	}
	
	/**
	 * save the prediction after convergence. 
	 * The format of the file is similar to format of check-ins file 
	 * @param fname	name of output file
	 * @throws IOException 
	 * @throws FileNotFoundException 
	 * @throws UnsupportedEncodingException 
	 */
	public void savePrediction(String fname) throws UnsupportedEncodingException, FileNotFoundException, IOException {
		ArrayList<String> result = new ArrayList<>();
		// parameters
		String parameters = "k=" + k + ";isSigmoid=" + isSigmoid + ";modeSim=" + modeSim + ";beta=" + beta + ";alpha=" + alpha
				+ ";isFriend=" + isFriend + ";lambda1=" + params.getLambda_1() + ";lambda2=" + params.getLambda_2() + ";lambda3=" 
				+ params.getLambda_3()	+ ";lambda_f=" + params.getLambda_f() + ";mu=" + mu;
		result.add(parameters);

		// user
		result.add("users:");
		for (String uId : userMap.keySet()) {
			StringBuffer sb = new StringBuffer();
			sb.append(uId + " ");
			UserObject uo = getUserObj(uId);
			sb.append(uo.getBias() + "," + Arrays.toString(uo.getFactors()));
			result.add(sb.toString());
		}
		
		// venue
		result.add("venues:");
		for (String vId : venueMap.keySet()) {
			StringBuffer sb = new StringBuffer();
			sb.append(vId + " ");
			VenueObject vo = getVenueObj(vId);
			sb.append(vo.getBias() + "," + Arrays.toString(vo.getEFactors()) + "," + Arrays.toString(vo.getIFactors()));
			result.add(sb.toString());
		}
		
		//write to file
		Utils.writeFile(result, fname);
	}
	
	public void test() {
		UserObject u = this.getUserObj("1380522");
//		VenueObject v = this.getVenueObj("4cb949aad78f4688e18cb573");
		
		u.setFactors(new double[]{0, 1.0, 2.0, 3.0, 64.0, 125.0, 6.0, 27.0, 108.0, 659.0});
//		v.setIFactors(new double[]{0, 1.0, 2.0, 3.0, 6124.0, 125.0, 6.0, 27.0, 18.0, 659.0});
		GradientCalculator g = new GradientCalculator(this, params);
				
		double[] gg = g.userGrad("1380522");
//		double[] gg = g.iVenueGrad("4b795c18f964a520b3f52ee3");
		for (int i = 0; i < k; i++)
			System.out.println(gg[i]);
		System.out.println("--------------");
		System.out.println(gg[4]);
		
		u.setFactors(new double[]{0, 1.0 , 2.0, 3.0, 64.0 + 0.0001, 125.0, 6.0, 27.0, 108.0, 659.0});
//		v.setIFactors(new double[]{0, 1.0, 2.0, 3.0, 6124.0 + 0.0001, 125.0, 6.0, 27.0, 18.0, 659.0});
		double f1 = 0.5 * objectiveFunc();
//		double f1 = 0.5 * calculateRMSE();
		
		u.setFactors(new double[]{0, 1.0 , 2.0, 3.0, 64.0 - 0.0001, 125.0, 6.0, 27.0, 108.0, 659.0});
//		v.setIFactors(new double[]{0, 1.0, 2.0, 3.0, 6124.0 - 0.0001, 125.0, 6.0, 27.0, 18.0, 659.0});
		double f2 = 0.5 * objectiveFunc();
//		double f2 = 0.5 * calculateRMSE();
		System.out.println( (f1 - f2) / (2 * 0.0001));
		
	}
	
	public static void main(String[] args) {
		System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", "13");
		
		// test gradient
//		String uFile = "UDI_full_Jakarta/full_user_profiles";
//		String vFile = "UDI_full_Jakarta/full_venue_profiles";
//		String nFile = "UDI_full_Jakarta/j_neighbor_10";
//		String fFile = null;
//		String cksFile = "UDI_full_Jakarta/full_cks.txt";
//		
////		String uFile = "UDI_full_Singapore/full_user_profiles";
////		String vFile = "UDI_full_Singapore/full_venue_profiles";
////		String nFile = "UDI_full_Singapore/s_neighbor_10";
////		String fFile = null;
////		String cksFile = "UDI_full_Singapore/full_cks.txt";
//		
//		boolean isSigmoid = false;
//		int modeSim = ModeSimilarity.COSIN_CKS_SIM;
////		int modeSim = ModeSimilarity.COSIN_DIST_SIM;
//		int k = 5;
//		double alpha = 0.1;
//		double beta = 0.1;
//		boolean isFriend = false;
//		Parameters params = new Parameters(0.01, 0.01, 0.01, 0.01);
//		Model m = new Model(uFile, vFile, nFile, fFile, cksFile, isSigmoid, modeSim, k, 
//						alpha, beta, isFriend, params);
////		m.test();
//		m.optimization();
		double[] x = new double[]{1, 2, 3};
		System.out.println(Arrays.toString(x));
	}
}