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

object BookRec {

  type BookRating = (Int, Double)
  type UserRatingPair = (String, (BookRating, BookRating))
  def filterDuplicates(userRatings:UserRatingPair):Boolean = {
    val bookRating1 = userRatings._2._1
    val bookRating2 = userRatings._2._2
    val bookISBN1 = bookRating1._1
    val bookISBN2 = bookRating2._1
    return (bookISBN1 == bookISBN2) && (bookRating1 == bookRating2)
  }

  def makePairs(userRatings:UserRatingPair) = {
    val bookRating1 = userRatings._2._1
    val bookRating2 = userRatings._2._2
    val book1 = bookRating1._1
    val rating1 = bookRating1._2
    val book2 = bookRating2._1
    val rating2 = bookRating2._2

    ((book1, book2), (rating1, rating2))
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

    // Map ratings to userID => (movie1, movie2)
    // userID => (rating1, rating2)
    val joinedISBNDF = ratingsDF.groupBy("userID").agg(collect_list("ISBN").as("ISBN"))
    val joinedRatingsDF = ratingsDF.groupBy("userID").agg(collect_list("rating").as("rating"))
    joinedISBNDF.show()
    joinedRatingsDF.show()

    spark.stop()
  }
}
