using System;
using System.Collections.Generic;
using System.Linq;
using Modeles_ML_POO;
using NumSharp;

namespace Test_Modeles
{
	class Program
	{
		static void Main(string[] args)
		{
			//TestMSELinearRegression();
			TestFit();
			Console.Read();
		}

		private static void TestLinearRegression()
		{
			var lrm = new LinearRegressionModel { Weights = np.array(new double[] { 2, 4, 6, 2, 2 }) };
			var x = np.array(new double[][] { new double[] { 1, 1, 1, 1 } });

			var list = LinearRegressionModel.MatrixToDataSet(x);

			var resultat = lrm.Predict(list);
			Console.WriteLine(resultat.ToString());
		}

		

		private static void TestMSELinearRegression()
		{
			var x = np.array(new double[,] { { 1, 1, 1, 1 }, { 2, 2, 2, 2 } });
			var weights = np.array(new double[] { 2, 2, 2, 2, 2 }) ;
			var y = np.array(new double[] { 3, 3});

			var resultat = LinearRegressionModel.MSELinearRegression(weights, x, y);
			Console.WriteLine(resultat.ToString());
		}

		private static void TestCostDerivative()
		{
			var x = np.array(new double[,] { { 1.95223362, 0.19171514 ,0.00512253, 0.80421507 }, { 2.33011483, 0.07497046, 2.37159478, 0.77167166 } });
			var weights = np.array(new double[] { 0.93176825, 1.25294411, 1.08222077 ,1.04574184, 1.54479269 });
			var y = np.array(new double[] { 2, 1 });

			
			var resultat = LinearRegressionModel.CostDerivative(weights, x, y);
			Console.WriteLine(resultat.ToString());
			
		}

		private static void TestGradientDescent()
		{
			var x = np.array(new double[,] { { 1.95223362, 0.19171514, 0.00512253, 0.80421507 }, { 2.33011483, 0.07497046, 2.37159478, 0.77167166 } });
			var y = np.array(new double[] { 2, 1 });
			var weight = np.array(np.random.rand(x.shape[1]+1)).reshape(x.shape[1] + 1, 1);
			var progress = LinearRegressionModel.GradientDescent(ref weight, x, y, 500, 0.1);
			
			//Console.WriteLine(np.array(progress.ToArray()).ToString());
			foreach (var d in progress)
			{
				Console.WriteLine(d);
			}

		}

		private static void TestFit()
		{
			var lrm = new LinearRegressionModel { Weights = np.array(new double[] { 2, 4, 6, 2, 2 }) };
			var x = np.array(new double[,] { { 1, 1, 1, 1 }, { 2, 5, 2, 2 }, { 2, 3, 2, 2 }, { 5, 2, 2, 2 }, { 2, 2, 3, 2 }, { 7, 2, 2, 2 }, { 2, 6, 2, 2 }, { 2, 4, 2, 2 }, { 2, 3, 3, 2 }, { 2, 2, 2, 8 } });

			var list = LinearRegressionModel.MatrixToDataSet(x);

			var yList = lrm.Predict(list);
			var y = np.array(yList.Cast<double>().ToArray());
			Console.WriteLine($"Weights {lrm.Weights.ToString()}");
			Console.WriteLine($"X {x.ToString()}");
			Console.WriteLine($"Y { y.ToString()}");

			lrm.Fit(list, yList);

			Console.WriteLine($"Weights after fit {lrm.Weights.ToString()}");

		}



	}
}
