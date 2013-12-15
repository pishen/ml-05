package core

import breeze.linalg.DenseVector
import math.pow
import scalax.io.Resource
import java.io.FileWriter
import scala.util.Random
import sys.process._

object Main {

  def main(args: Array[String]): Unit = {
    val trainLines = Resource.fromFile("hw5_14_train.dat").lines().map(line => {
      val tokens = line.split(" ").filter(_ != "")
      val label = tokens.last.toInt
      val features = tokens.init.map(_.toDouble).zipWithIndex.filter(_._1 > 0.0).map{
        case (f, i) => (i + 1) + ":" + f
      }.mkString(" ")
      label + " " + features
    })
    Resource.fromWriter(new FileWriter("train")).writeStrings(trainLines, "\n")
    
    val gammas = Seq(0.125, 0.5, 2.0).map(s => 1.0 / (2 * pow(s, 2)))
    val costs = Seq(0.001, 1.0, 1000.0)
    val gamma = 1.0 / (2 * pow(0.125, 2))
    val cost = 0.001
    println("train")
    Seq("./svm-train", "-c", cost.toString, "-g", gamma.toString, "train", "train.m").!
    println("cv")
    Seq("./svm-train", "-c", cost.toString, "-g", gamma.toString, "-v", "5", "train").!
  }

  def pb13() = {
    def processData(lines: Seq[String]) = {
      lines.map(_.split(" ")).map(tokens => (DenseVector(tokens.init.map(_.toDouble)), tokens.last.toInt))
    }

    def solveQP(data: Seq[(DenseVector[Double], Int)]) = {
      val d = data.head._1.length
      //notation of octave QP
      val H = Seq.tabulate(d + 1, d + 1)((i, j) => if (i == j && i > 0 && j > 0) 1 else 0)
      Resource.fromWriter(new FileWriter("octave-input")).writeStrings(Seq(
        "x0 = [" + Seq.fill(d + 1)(0).mkString(";") + "]",
        "H = [" + H.map(_.mkString(",")).mkString(";") + "]",
        "q = [" + Seq.fill(d + 1)(0).mkString(";") + "]",
        "A = []",
        "b = []",
        "lb = []",
        "ub = []",
        "A_lb = [" + Seq.fill(data.length)(1).mkString(";") + "]",
        "A_in = [" + data.map { case (x, y) => y + "," + x.toArray.map(_ * y).mkString(",") }.mkString(";") + "]",
        "A_ub = []",
        "qp(x0,H,q,A,b,lb,ub,A_lb,A_in,A_ub)"), "\n")

      val res = "octave octave-input".!!

      val u = res.split("ans =").last.split("\\s+").filter(_ != "").map(_.toDouble)
      //println("u=" + u.mkString(","))
      (u.head, DenseVector(u.tail))
    }

    val train = processData(Resource.fromFile("hw5_13_train.dat").lines().toSeq)
    val test = processData(Resource.fromFile("hw5_13_test.dat").lines().toSeq)

    val sampleSize = (train.size * 0.8).toInt
    val points = for (i <- 1 to 100) yield {
      val sampleTrain = Random.shuffle(train).take(sampleSize)
      val (b, w) = solveQP(sampleTrain)
      val margin = 1 / w.norm(2)
      val eout = test.count { case (x, y) => (w.dot(x) + b) * y < 0 } / test.length.toDouble
      println("iter " + i + ": " + margin + ", " + eout)
      margin + " " + eout
    }
    Resource.fromWriter(new FileWriter("points")).writeStrings(points, "\n")
    println("exit: " + "gnuplot gnuplot-script-13".!)
  }

  def pb1to5() = {
    val rawX = Seq((1, 0), (0, 1), (0, -1), (-1, 0), (0, 2), (0, -2), (-2, 0))
    val rawY = Seq(-1, -1, -1, 1, 1, 1, 1)
    val x = rawX.map(raw => DenseVector[Double](raw._1, raw._2)).toArray
    val y = DenseVector[Double](rawY.map(_.toDouble).toArray)
    def K(a: DenseVector[Double], b: DenseVector[Double]) = pow(2 + a.dot(b), 2)

    val H = Seq.tabulate(7, 7)((i, j) => y(i) * y(j) * K(x(i), x(j)))
    Resource.fromWriter(new FileWriter("octave-input")).writeStrings(Seq(
      "x0 = [" + Seq.fill(7)(0).mkString(";") + "]",
      "H = [" + H.map(_.mkString(",")).mkString(";") + "]",
      "q = [" + Seq.fill(7)(-1).mkString(";") + "]",
      "A = [" + rawY.mkString(",") + "]",
      "b = [0]",
      "lb = [" + Seq.fill(7)(0).mkString(";") + "]",
      "ub = []",
      "qp(x0,H,q,A,b,lb,ub)"), "\n")

    val res = "octave octave-input".!!

    val rawAlpha = res.split("ans =").last.split("\\s+").filter(_ != "").map(_.toDouble)
    println("alpha: " + rawAlpha.mkString(", "))
    val alpha = DenseVector[Double](rawAlpha)

    val m = rawAlpha.toSeq.indexWhere(_ > 0.0)
    val indices = rawAlpha.indices
    val b = y(m) - indices.map(i => alpha(i) * y(i) * K(x(i), x(m))).sum

    val c = alpha.dot(y) * 4 + b
    val x1 = indices.map(i => alpha(i) * y(i) * 4 * x(i)(0)).sum
    val x2 = indices.map(i => alpha(i) * y(i) * 4 * x(i)(1)).sum
    val x1s = indices.map(i => alpha(i) * y(i) * x(i)(0) * x(i)(0)).sum
    val x1x2 = indices.map(i => alpha(i) * y(i) * 2 * x(i)(0) * x(i)(1)).sum
    val x2s = indices.map(i => alpha(i) * y(i) * x(i)(1) * x(i)(1)).sum
    println("c: " + c)
    println("x1: " + x1)
    println("x2: " + x2)
    println("x1s: " + x1s)
    println("x1x2: " + x1x2)
    println("x2s: " + x2s)

    rawAlpha.zipWithIndex.filter(_._1 > 0.0).map(_._2)
      .map(m => y(m) - indices.map(i => alpha(i) * y(i) * K(x(i), x(m))).sum)
      .foreach(b => println("b: " + b))

    indices.foreach(m => println("distance of " + m + ": " + (y(m) - indices.map(i => alpha(i) * y(i) * K(x(i), x(m))).sum - b)))
  }

}