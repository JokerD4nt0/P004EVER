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

		private NDArray RunLinearRegression(NDArray x, NDArray weight)
		{
			//On teste s'il y a un biais et donc une coordonnée constante à 1 à rajouter à la fin de x
			if (weight.shape[0] == x.shape[1]+1)
			{
				var vect_init = np.ones((x.shape[0]), 1);
				x = np.concatenate((vect_init, x), 1);
			}

			var toReturn = np.matmul(x, weight);
			return toReturn;
		}



	}
}