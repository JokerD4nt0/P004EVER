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
			var lrm = new LinearRegressionModel {Weights = np.array(new double[] {2, 4, 6, 2, 2})};
			var x = np.array(new double[][] {new double[] {1, 1, 1, 1}});
			
			//Transformation de X depuis une matrice vers une collection de collection d'objets
			var array = x.ToJaggedArray<double>();
			var list = array.Cast<double[]>().Select(point=>new List<object>(point.Cast<object>())).Cast<ICollection<object>>().ToArray();

			var resultat = lrm.Predict(list);
			Console.WriteLine(resultat.ToString());
			Console.Read();
		}
	}
}
