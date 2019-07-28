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

  /*
   * @param {list} args - the list of ISBNs you read
   */
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val sc = new SparkContext("local[*]", "BookSimilarities")

    val data = sc.textFile("./data/sample.csv")

  }
}
