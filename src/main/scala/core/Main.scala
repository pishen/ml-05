package core

import breeze.linalg.DenseVector
import breeze.linalg.diag
import breeze.linalg.inv
import math.pow
import math.exp
import scalax.io.Resource
import java.io.FileWriter
import scala.util.Random
import sys.process._
import breeze.linalg.DenseMatrix

object Main {

  def main(args: Array[String]): Unit = {
    pb14to16()
  }

  def pb14to16() = {
    val trainLines = Resource.fromFile("hw5_14_train.dat").lines().map(line => {
      val tokens = line.split(" ").filter(_ != "")
      val label = tokens.last.toInt
      val features = tokens.init.map(_.toDouble).zipWithIndex.filter(_._1 != 0.0).map {
        case (f, i) => (i + 1) + ":" + f
      }.mkString(" ")
      label + " " + features
    }).toSeq
    Resource.fromWriter(new FileWriter("train")).writeStrings(trainLines, "\n")
    val tSize = trainLines.size

    val sigmas = Seq(0.025, 0.5, 2.0)
    val costs = Seq(0.001, 1.0, 1000.0)
    lazy val pb14 = sigmas.flatMap(sigma => costs.map(cost => (sigma, cost))).map {
      case (sigma, cost) => {
        val gamma = 1.0 / (2 * pow(sigma, 2))
        val nSVN = Seq("./svm-train", "-c", cost.toString, "-g", gamma.toString, "train", "train.m").!!
          .split("\n").last.split(" ").last.toInt / tSize.toDouble
        val ein = 1 - (Seq("./svm-predict", "train", "train.m", "predict").!!
          .split("\n").last.split(" ").find(_.contains("%")).get.init.toDouble / 100)
        val ecv = 1 - (Seq("./svm-train", "-c", cost.toString, "-g", gamma.toString, "-v", "5", "train").!!
          .split("\n").last.split(" ").last.init.toDouble / 100)
        TestCase(sigma, cost, nSVN, ein, ecv)
      }
    }
    lazy val pb15 = sigmas.flatMap(sigma => costs.map(cost => (sigma, cost))).map {
      case (sigma, cost) => {
        val gamma = 1.0 / (2 * pow(sigma, 2))
        val svmCmd = Seq("./svm-train", "-s", "3", "-p", "0.01", "-c", cost.toString, "-g", gamma.toString)
        val nSVN = (svmCmd ++ Seq("train", "train.m")).!!
          .split("\n").last.split("[, ]")(2).toInt / tSize.toDouble
        val ein = {
          assert(Seq("./svm-predict", "train", "train.m", "predict").! == 0)
          val predicts = Resource.fromFile("predict").lines().map(l => if (l.toDouble > 0.0) 1 else -1).toSeq
          val ys = trainLines.map(_.split(" ").head.toInt)
          predicts.zip(ys).count(p => p._1 != p._2) / tSize.toDouble
        }
        val ecv = {
          //TODO shuffle?
          val subsets = trainLines.grouped((trainLines.size / 5.0).ceil.toInt).toSeq
          subsets.indices.map(i => {
            val vld = subsets(i)
            val subTrain = subsets.zipWithIndex.filter(_._2 != i).map(_._1).reduce(_ ++ _)
            Resource.fromWriter(new FileWriter("sub-train")).writeStrings(subTrain, "\n")
            assert((svmCmd ++ Seq("sub-train", "sub-train.m")).! == 0)
            Resource.fromWriter(new FileWriter("validate")).writeStrings(vld, "\n")
            assert(Seq("./svm-predict", "validate", "sub-train.m", "predict").! == 0)
            val predicts = Resource.fromFile("predict").lines().map(l => if (l.toDouble > 0.0) 1 else -1).toSeq
            val ys = vld.map(_.split(" ").head.toInt)
            predicts.zip(ys).count(p => p._1 != p._2) / predicts.size.toDouble
          }).sum / subsets.size
        }
        TestCase(sigma, cost, nSVN, ein, ecv)
      }
    }
    lazy val pb16 = {
      val train = Resource.fromFile("hw5_14_train.dat").lines().toSeq.map(line => {
        val tokens = line.split(" ").filter(_ != "")
        val x = DenseVector(tokens.init.map(_.toDouble))
        val y = tokens.last.toInt
        (x, y)
      })
      sigmas.flatMap(sigma => costs.map(cost => (sigma, cost))).map {
        case (sigma, lambda) => {
          def kernel(xn: DenseVector[Double], xm: DenseVector[Double]) = {
            val dis = xn - xm
            exp(-1 * dis.dot(dis) / (2 * sigma * sigma))
          }
          val K = DenseMatrix.tabulate(tSize, tSize)((n, m) => kernel(train(n)._1, train(m)._1))
          val yv = DenseVector(train.map(_._2.toDouble).toArray)
          val beta = inv(diag(DenseVector.fill(tSize)(lambda)) + K) * yv
          val nSVN = beta.toArray.count(_ != 0.0) / tSize.toDouble
          val ein = train.count {
            case (x, y) => train.indices.map(i => beta(i) * kernel(train(i)._1, x)).sum * y < 0.0
          } / tSize.toDouble
          val ecv = {
            val subsets = train.grouped((train.size / 5.0).ceil.toInt).toSeq
            subsets.indices.map(i => {
              val vld = subsets(i)
              val subTrain = subsets.zipWithIndex.filter(_._2 != i).map(_._1).reduce(_ ++ _)
              val subK = DenseMatrix.tabulate(subTrain.size, subTrain.size) {
                (n, m) => kernel(subTrain(n)._1, subTrain(m)._1)
              }
              val subYV = DenseVector(subTrain.map(_._2.toDouble).toArray)
              val subBeta = inv(diag(DenseVector.fill(subTrain.size)(lambda)) + subK) * subYV
              vld.count {
                case (x, y) => subTrain.indices.map(i => subBeta(i) * kernel(subTrain(i)._1, x)).sum * y < 0.0
              } / vld.size.toDouble
            }).sum / subsets.size.toDouble
          }
          TestCase(sigma, lambda, nSVN, ein, ecv)
        }
      }
    }

    def writeRes(s: Seq[TestCase], filename: String) = {
      val res = s.flatMap(t => {
        "================" ::
          "sigma: " + t.sigma ::
          "cost: " + t.cost ::
          "Ein: " + t.ein ::
          "Ecv: " + t.ecv ::
          "nSV/N: " + t.nSVN ::
          Nil
      })
      Resource.fromWriter(new FileWriter(filename)).writeStrings(res, "\n")
    }
    writeRes(pb14, "pb14")
    //writeRes(pb15, "pb15")
    //print pb16 result
    /*val res = pb16.flatMap(t => {
      "================" ::
        "sigma: " + t.sigma ::
        "lambda: " + t.cost ::
        "Ein: " + t.ein ::
        "Ecv: " + t.ecv ::
        "nSV/N: " + t.nSVN ::
        Nil
    })
    Resource.fromWriter(new FileWriter("pb16")).writeStrings(res, "\n")*/
  }

  case class TestCase(sigma: Double, cost: Double, nSVN: Double, ein: Double, ecv: Double)

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