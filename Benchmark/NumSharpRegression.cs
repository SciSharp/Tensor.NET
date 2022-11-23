using BenchmarkDotNet.Attributes;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LinearRegression
{
    public class NumSharpRegressionTest
    {
        public NDArray data;
        public NDArray label;
        public static double splitRate = 0.8;
        public static int recordInterval = 10;
        Model model;
        [GlobalSetup]
        public void Setup()
        {
            (data, label) = IrisLoader.Load(@"");
            model = new Model(BenchmarkParam.FeatureCount);
            model.InitializeParameters();
        }
        [Benchmark]
        public void TrainNumSharp()
        {
            model.Train(data, label, BenchmarkParam.TrainEpochs, 0.001, recordInterval);
        }
        public static class IrisLoader
        {
            public static Dictionary<string, double> mapping;
            static IrisLoader()
            {
                mapping = new Dictionary<string, double>();
                mapping.Add("Iris-setosa", 0);
                mapping.Add("Iris-versicolor", 1);
                mapping.Add("Iris-virginica", 2);
            }
            public static (NDArray, NDArray) Load(string path)
            {
                NDArray data = np.random.normal(0, 2, new int[] { BenchmarkParam.FeatureCount, BenchmarkParam.DataLength });
                NDArray label = np.random.normal(0, 1, new int[] { 1, BenchmarkParam.DataLength });
                return (data, label);
            }
        }
        public static class Sigmoid
        {
            public static NDArray Run(NDArray src)
            {
                return 1 / (1 - src);
            }
        }

        public class Model
        {
            private int _dataDim;
            public NDArray w;
            public double b;
            public Model(int dataDim)
            {
                _dataDim = dataDim;
                InitializeParameters();
            }
            public (NDArray, double) InitializeParameters()
            {
                w = np.random.normal(0, 0.01f, new int[] { _dataDim, 1 }).astype(typeof(double));
                b = 0;
                return (w, b);
            }
            public (NDArray, NDArray, NDArray) ForwardAndBackwardPropagate(NDArray data, NDArray label)
            {
                var dataNum = data.shape[0];
                // forward propagation
                var z = w.transpose(new int[] { 0, 1 }) * data + b;
                var predict = Sigmoid.Run(z);
                var diff = predict - label;

                var cost = np.mean((label * np.log2(predict) + (1 - label) * np.log2(1 - predict)) * (-1));

                // back propagation
                var dw = data * diff.transpose(new int[] { 0, 1 }) / dataNum;
                var db = Sum(diff) / dataNum;
                return (cost, dw, db);
            }

            private double Sum(NDArray src)
            {
                double res = 0;
                for (int i = 0; i < src.shape[0]; i++)
                {
                    for (int j = 0; j < src.shape[1]; j++)
                    {
                        res += src.GetDouble(i, j);
                    }
                }
                return res;
            }

            public NDArray UpdataParameters(NDArray data, NDArray label, double lr)
            {
                var (cost, gradW, gradB) = ForwardAndBackwardPropagate(data, label);

                w -= lr * gradW;
                b -= lr * gradB.GetDouble(0);

                return cost;
            }

            public List<double> Train(NDArray data, NDArray label, int epochs, double lr, int recordInterval = 5)
            {
                var costs = new List<double>();
                for (int i = 1; i <= epochs; i++)
                {
                    var cost = UpdataParameters(data, label, lr);
                    if (i % recordInterval == 0)
                    {
                        costs.Add(cost.GetDouble(0));
                    }
                }
                return costs;
            }
        }
    }
}
