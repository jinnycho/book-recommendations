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

  type BookRating = (String, Double)
  type UserRatingPair = (String, (BookRating, BookRating))
  def filterDuplicates(userRatings:UserRatingPair):Boolean = {
    val bookRating1 = userRatings._2._1
    val bookRating2 = userRatings._2._2

    val book1 = bookRating1._1
    val book2 = bookRating2._1
    return book1 != book2
  }

  /*
   * @param {list} args - the list of ISBNs you read
   */
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val sc = new SparkContext("local[*]", "BookSimilarities")

    val data = sc.textFile("./data/sample.csv")
    // userID => ISBN, rating
    val ratings = data.map(l => l.split(",")).map(l => (l(1), (l(2), l(3).toDouble)))
    // Find every pair of books that were read by the same person
    val joinedRatings = ratings.join(ratings)
    // Filter out duplicate pairs
    val uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)
  }
}
