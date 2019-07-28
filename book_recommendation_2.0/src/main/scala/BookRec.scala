package com.jinnycho503.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import scala.math.sqrt
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.feature.Normalizer

object BookRec {

  def computeCosineSim(userInfoDF: DataFrame) = {
    // Compute TF-IDF
    val hashingTF = new HashingTF()
      .setInputCol("rating")
      .setOutputCol("rawFeatures")
    val tf = hashingTF.transform(userInfoDF)
    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
    val idfModel = idf.fit(tf)
    val rescaledData = idfModel.transform(tf)
    // Compute L2 norm
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)
    val l1NormData = normalizer.transform(rescaledData)
    l1NormData.show()
  }

  /*
   * @param {list} args - the list of ISBNs you read
   */
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .appName("Book Recommendations")
      .master("local[*]")
      .getOrCreate()

    val customSchema = StructType(Array(
      StructField("index", IntegerType, true),
      StructField("userID", StringType, true),
      StructField("ISBN", StringType, true),
      StructField("rating", DoubleType, true)))

    val ratingsDF = spark.read
      .format("csv")
      .option("header", "true")
      .option("charset", "UTF8")
      .schema(customSchema)
      .load("./data/sample.csv")

    // Map data to (userID), (movie1, movie2,...)
    val joinedISBNDF = ratingsDF.groupBy("userID").agg(collect_list("ISBN").as("ISBN"))
    // (userID), (rating1, rating2,...)
    val joinedRatingsDF = ratingsDF.groupBy("userID").agg(collect_list("rating").as("rating"))

    // Create DF (userID), (movie1, movie2...), (rating1, rating2...)
    val userInfoJoinedDF = joinedISBNDF
      .join(joinedRatingsDF, joinedISBNDF("userID") === joinedRatingsDF("userID"), "left_outer")
      .select(joinedISBNDF("userID"), joinedISBNDF("ISBN"), joinedRatingsDF("rating"))

    // Compute similarities
    // returns a new dataframe:
    // (userID1), (userID2), (movie1, movie2...), (rating1, rating2), (sim)
    val moviePairSims = computeCosineSim(userInfoJoinedDF)
    spark.stop()
  }
}
