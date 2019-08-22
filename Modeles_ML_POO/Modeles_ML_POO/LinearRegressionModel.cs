using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace Modeles_ML_POO
{
	public class LinearRegressionModel : IModel
	{
		public NDArray Weights { get; set; } 

		public void Fit(ICollection<ICollection<object>> featuresDataset, ICollection<object> labels)
		{
			if (featuresDataset == null || featuresDataset.Count==0)
			{
				throw new ApplicationException("Dataset must be non null");
			}
			int nbItemsDataSet = featuresDataset.Count;
			NDArray data = np.arange(nbItemsDataSet * featuresDataset.First().Count).reshape(nbItemsDataSet, featuresDataset.First().Count);

			


		}

		//public NDArray TransformPoint(IEnumerable<object> point)
		//{

		//}

		public object Predict(ICollection<ICollection<object>> dataPoints)
		{
			//Utilisation d'une méthode et d'un délégué
			//IEnumerable<NDArray> arrayPoints = dataPoints.Select(TransformPoint).ToArray();


			//utilisation d'une expression lambda
			IEnumerable<double[]> arrayPoints = dataPoints.Select(point => (point.Select(Convert.ToDouble).ToArray()));

			var x = np.array(arrayPoints.ToArray());

			var toReturn = RunLinearRegression(x, Weights);
			return toReturn;
		}

		public static NDArray RunLinearRegression(NDArray x, NDArray weight)
		{
			//On teste s'il y a un biais et donc une coordonnée constante à 1 à rajouter à la fin de x
			if (weight.shape[0] == x.shape[1]+1)
			{
				x = AddOnes(x);
			}

			var toReturn = np.matmul(x, weight).flatten();
			return toReturn;
		}

		private static NDArray AddOnes(NDArray x)
		{
			var vect_init = np.ones((x.shape[0]), 1);
			x = np.concatenate((vect_init, x), 1);
			return x;
		}

		public static double MSELinearRegression(NDArray weight, NDArray x, NDArray y)
		{
			var h = RunLinearRegression(x, weight);
			var erreur = h - y;
			var mse = (1.0 / (2*x.shape[0])) * (np.sum(erreur*erreur));
			return mse.GetDouble();
		}

		public static NDArray CostDerivative(NDArray weight, NDArray x, NDArray y)
		{
			if (weight.shape[0] == x.shape[1] + 1)
			{
				x = AddOnes(x);
			}
			var deriv = new List<double>();
			for (int i = 0; i < weight.shape[0]; i++)
			{
				var h = RunLinearRegression(x, weight);
				var erreur = h - y;
				var erreurX = x[Slice.All, i] * erreur;
				var der = np.sum(erreurX) / y.shape[0];
				deriv.Add(der);
			}

			var toReturn = np.array(deriv.ToArray()).reshape(weight.shape[0], 1);

			return toReturn;

		}

		public static IEnumerable<double> GradientDescent(ref NDArray weight, NDArray x, NDArray y, int maxIterations, double alpha)
		{
			if (weight.shape[0] == x.shape[1] + 1)
			{
				x = AddOnes(x);
			}

			var progress = new List<double>();
			double coutm1 = 1;
			for (int i = 0; i < maxIterations; i++)
			{
				var derived = CostDerivative(weight, x, y);
				var alphaDeriv = alpha * derived;
				weight = weight - alphaDeriv;
				
				var cout = MSELinearRegression(weight, x, y);

				progress.Add(cout);
				
				//alpha = Math.Max(alpha*0.999, alpha * cout);
				alpha = alpha * (1 + cout - coutm1);

				coutm1 = cout;
			}

			return progress;
		}
	}
}