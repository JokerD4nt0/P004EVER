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
			TestCostDerivative();
			Console.Read();
		}

		private static void TestLinearRegression()
		{
			var lrm = new LinearRegressionModel { Weights = np.array(new double[] { 2, 4, 6, 2, 2 }) };
			var x = np.array(new double[][] { new double[] { 1, 1, 1, 1 } });

			//Transformation de X depuis une matrice vers une collection de collection d'objets
			var array = x.ToJaggedArray<double>();
			var list = array.Cast<double[]>().Select(point => new List<object>(point.Cast<object>())).Cast<ICollection<object>>().ToArray();

			var resultat = lrm.Predict(list);
			Console.WriteLine(resultat.ToString());
		}

		private static void TestMSELinearRegression()
		{
			var x = np.array(new double[,] { { 1, 1, 1, 1 }, { 2, 2, 2, 2 } });
			var weights = np.array(new double[] { 2, 2, 2, 2, 2 }) ;
			var y = np.array(new double[] { 3, 3});

			var lrm = new LinearRegressionModel();
			var resultat = lrm.MSELinearRegression(weights, x, y);
			Console.WriteLine(resultat.ToString());
		}

		private static void TestCostDerivative()
		{
			var x = np.array(new double[,] { { 1.95223362, 0.19171514 ,0.00512253, 0.80421507 }, { 2.33011483, 0.07497046, 2.37159478, 0.77167166 } });
			var weights = np.array(new double[] { 0.93176825, 1.25294411, 1.08222077 ,1.04574184, 1.54479269 });
			var y = np.array(new double[] { 2, 1 });

			var lrm = new LinearRegressionModel(){Weights = weights};
			
			var resultat = lrm.CostDerivative(weights, x, y);
			Console.WriteLine(resultat.ToString());
			
		}



	}
}
