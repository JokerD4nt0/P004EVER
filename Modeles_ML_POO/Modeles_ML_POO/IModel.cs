using System;
using System.Collections.Generic;

namespace Modeles_ML_POO
{
    public interface IModel
    {

	    void Fit(ICollection<ICollection<object>> featuresDataset, ICollection<object> labels);

	    ICollection<object> Predict(ICollection<ICollection<object>> dataPoints);

	}
}
