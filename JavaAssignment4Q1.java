package javaassignment4;

// Assignment 4 :   Tools & Techniques for Large Scale Data Analytics
// Name         :   Rahul Kumar
// Student ID   :   20230113

// Q1. Cluster given Twitter tweets by their geographic origins (coordinates), using
// the K-means clustering algorithm.

// Importing the spark packages
// Importing spark mllib packages

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.commons.lang.StringUtils;
import org.apache.spark.api.java.JavaPairRDD;
import scala.Tuple2;

// class JavaAssignment4Q1 containing the main()
public class JavaAssignment4Q1 {
    // Driver function defination
    public static void main(String[] args) {
        
        // setting up the Spark configuration and Spark context
        System.setProperty("hadoop.home.dir", "C:/winutils");
        SparkConf sparkConf =   new SparkConf()
                                    .setAppName("Twitter_Clusters")
                                    .setMaster("local[4]")
                                    .set("spark.executor.memory", "1g");

        JavaSparkContext ctx = new JavaSparkContext(sparkConf);
        
        
        System.out.println("\n\n \t\t\t\t Assignment4Q1\n\n\n ");
        
        // path where the txt file is stored
        String path = "twitter2D.txt";
        
        // Reading the twitter txt file in JavaRDD
        JavaRDD<String> text = ctx.textFile(path, 1);
        
        // Creating JavaRDD of all the co-ordinates by mapping each line to a vector of coordinates(1st 2 columns)
        JavaRDD<Vector> co_ordinates = text.map(line -> {
            String[] split_array = line.split(",");         // split each line at " , "
            double[] values = new double[2];                // create a array of doubles of size 2 - [coordinate 1, coordinate 2]
            for (int i = 0; i < 2; i++) {                   // filling in the coordinates
                values[i] = Double.parseDouble(split_array[i]);
            }
            return Vectors.dense(values);                   // return a Vector of coordinates
        });
        
        // cache training data as needed to build the model
        co_ordinates.cache();                                 

        // Cluster the data into 4 classes using KMeans
        int numClusters = 4;
        int numIterations = 1000;
        // train the model on the features (coordinates)
        KMeansModel clusters = KMeans.train(co_ordinates.rdd(), numClusters, numIterations);
        
        
        // creating a JavaPairRDD with each line in twitter text file mapped to 
        // pair (cluster the tweet belongs to (depending on trained model), actual-tweet)
        
        JavaPairRDD<Integer,String> tweet_data = text.mapToPair(tweet -> {
            String[] tweet_element = tweet.split(",");      // split each line at " , "
            String final_tweet = "";                        // variable to store the actual tweet
            int position = 1;                               // variable to find the starting position of the actual tweet
            double[] values = new double[2];                // create a array of doubles of size 2 - [coordinate 1, coordinate 2]
            	
            for (int i = 0; i < 2; i++) {                   // filling in the coordinates
                    values[i] = Double.parseDouble(tweet_element[i]);
            }
            
            Vector V = Vectors.dense(values);               // assigning my vector of coordinates to variable V
            
            // loop to find the starting position of the actual tweet 
            // if a tweet contained "," ==> splitting by "," would have seperated the actual tweet into seperate elements
            //
            // for position = 1 till length of the variable(tweet_element) after split
            // check (variable.length - 1) to be numeric, if false repeat the loop
            //                                            if true, break the loop
            //
            // working example : S = [1,2,3,Hi,there].
            // length of S = 5 (0-4 indexes)
            // for position = 1; S.length - position = 4, check S[4] to be numeric ==> false, continue with next iteration.
            // for position = 2; S.length - position = 3, check S[3] to be numeric ==> false, continue with next iteration.
            // for position = 3; S.lenght - position = 2, check S[2] to be numeric ==> true, break the loop.
            
            for(position = 1; position < tweet_element.length; position++){
                if(StringUtils.isNumeric(tweet_element[tweet_element.length - position]) == true){
                    break;
                }
            }
            
            // run loop from position-1(actual starting position) till last position and append it to empty string to get the actual tweet
            // Continued example: position = 3 (from above)
            // j = position-1 ==> 2; final_tweet = "" + S[S.length - j] ==> S[3]  + "," ==> final_tweet = "" + "Hi" + ","  ==> "Hi,"
            // j = position-1 ==> 1; final_tweet = "Hi," + S[S.length - j] ==> S[4] ==> final_tweet = "Hi," + "there" ==> "Hi,there"
            
            for(int j = position-1; j > 0; j-- ){
                if(j==1)
                    final_tweet = final_tweet + tweet_element[tweet_element.length - j];
                else
                    final_tweet = final_tweet + tweet_element[tweet_element.length - j] + "," ;
            }    
            
            // return tuple(key,value) pair as (cluster the tweet belongs to (depending on trained model), actual-tweet)
            return new Tuple2<Integer,String>(clusters.predict(V),final_tweet);
        });
        
        // sorting the above JavaPairRDD by key value in ascending order
        JavaPairRDD<Integer,String> tweet_data_sorted = tweet_data.sortByKey(true);
        
        // print the tweet in each cluster in sortedRDD
        tweet_data_sorted.foreach(result -> {
            System.out.println("\n\n Tweet \t\"" + result._2() + "\" is in cluster: " + result._1()+ "\n\n");
        });
        
        // stopping the Spark context object and closing it.
        ctx.stop();
        ctx.close();   
    }
}


  