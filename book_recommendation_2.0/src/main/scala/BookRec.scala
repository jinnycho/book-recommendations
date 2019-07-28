package com.jinnycho503.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import scala.math.sqrt

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

  def makePairs(userRatings:UserRatingPair) = {
    val bookRating1 = userRatings._2._1
    val bookRating2 = userRatings._2._2

    val book1 = bookRating1._1
    val rating1 = bookRating1._2
    val book2 = bookRating2._1
    val rating2 = bookRating2._2

    ((book1, book2), (rating1, rating2))
  }

  type RatingPair = (Double, Double)
  type RatingPairs = Iterable[RatingPair]
  def computeCosineSim(ratingPairs:RatingPairs): (Double, Int) = {
    var numPairs:Int = 0
    var sum_11:Double = 0.0
    var sum_22:Double = 0.0
    var sum_12:Double = 0.0

    for (pair <- ratingPairs) {
      val rating1 = pair._1
      val rating2 = pair._2
      sum_11 += rating1 * rating1
      sum_22 += rating2 * rating2
      sum_12 += rating1 * rating2
      numPairs += 1
    }

    val numerator:Double = sum_12
    val denominator = sqrt(sum_11) * sqrt(sum_22)
    var score:Double = 0.0
    if (denominator != 0) {
      score = numerator/denominator
    }
    return (score, numPairs)
  }

  /*
   * @param {list} args - the list of ISBNs you read
   */
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val sc = new SparkContext("local[*]", "BookSimilarities")

    val data = sc.textFile("./data/AmazonSorted.csv")
    // userID => ISBN, rating
    val ratings = data.map(l => l.split(",")).map(l => (l(1), (l(2), l(3).toDouble)))
    // Find every pair of books that were read by the same person
    val joinedRatings = ratings.join(ratings)
    // Filter out duplicate pairs
    val uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)
    // make the book pairs the key
    val bookPairs = uniqueJoinedRatings.map(makePairs)
    // Collect all ratings for each book pair
    // (movie1, movie2) => (rating1, rating2), (rating1, rating2)...
    val bookPairRatings = bookPairs.groupByKey()

    // compute similarities
    val bookPairSims = bookPairRatings.mapValues(computeCosineSim).cache()

    if (args.length > 0) {
      val simThreshold = 0.98
      val occurenceThreshold = 30.0
      val bookID:Int = args(0).toInt

      val filteredResults = bookPairSims.filter(x => {
        val bookPair = x._1
        val sim = x._2
        (bookPair._1 == bookID || bookPair._2 == bookID) && sim._1 > simThreshold && sim._2 > occurenceThreshold
      })

      // sort by quality score
      val results = filteredResults
        .map(x => (x._2, x._1))
        .sortByKey(false)
        .take(10)

      for (result <- results) {
        val sim = result._1
        val bookPair = result._2
        var similarBookID = bookPair._1
        if (similarBookID == bookID) {
          similarBookID = bookPair._2
        }
        //TODO remove redundant recommendations
        //TODO use API to return the actual book name
        println("Recommends: " + similarBookID + "\tscore: " + sim._1 + "\tstrength: " + sim._2)
      }
    }
  }
}
