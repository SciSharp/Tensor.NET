using BenchmarkDotNet.Attributes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NN.Core;

namespace LinearRegression
{
    public class NativeArrayRegressionTest
    {
        public NDArray<double> data;
        public NDArray<double> label;
        public Model model;
        public static double splitRate = 0.8;
        public static int recordInterval = 10;

        [GlobalSetup]
        public void Setup()
        {
            (data, label) = IrisLoader.Load(@"");
            model = new Model(BenchmarkParam.FeatureCount);
            model.InitializeParameters();
        }

        [Benchmark]
        public void TrainGenericMath()
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
            public static (NDArray<double>, NDArray<double>) Load(string path)
            {
                var data = NDArray<double>.Random.Normal(new int[] { BenchmarkParam.FeatureCount, BenchmarkParam.DataLength }, 0, 2);
                var label = NDArray<double>.Random.Normal(new int[] { 1, BenchmarkParam.DataLength }, 0, 1);
                return (data, label);
            }
        }
        public static class Sigmoid
        {
            public static NDArray<double> Run(NDArray<double> src)
            {
                return 1 / (1 + NDArray<double>.Pow(Math.E, -src));
            }
        }

        public class Model
        {
            private int _dataDim;
            public NDArray<double> w;
            public double b;
            public Model(int dataDim)
            {
                _dataDim = dataDim;
                InitializeParameters();
            }
            public (NDArray<double>, double) InitializeParameters()
            {
                w = NDArray<double>.Random.Normal(new int[] { _dataDim, 1 }, 0, 0.01);
                b = 0;
                return (w, b);
            }
            public (NDArray<double>, NDArray<double>, NDArray<double>) ForwardAndBackwardPropagate(in NDArray<double> data, in NDArray<double> label)
            {
                var dataNum = data.Shape[0];
                // forward propagation
                var z = w.Transpose(0, 1) * data + b;
                var predict = Sigmoid.Run(z);
                var diff = predict - label;

                var cost = (-(label.Multiply(NDArray<double>.Log2(predict)) + (1 - label).Multiply(NDArray<double>.Log2(1 - predict)))).Mean();

                // back propagation
                var dw = data * diff.Transpose(0, 1) / dataNum;
                var db = diff.Sum() / dataNum;
                return (cost, dw, db);
            }

            public NDArray<double> UpdateParameters(NDArray<double> data, NDArray<double> label, double lr)
            {
                var (cost, gradW, gradB) = ForwardAndBackwardPropagate(data, label);

                w -= lr * gradW;
                b -= lr * gradB[0, 0];

                return cost;
            }

            public List<double> Train(NDArray<double> data, NDArray<double> label, int epochs, double lr, int recordInterval = 5)
            {
                var costs = new List<double>();
                for (int i = 1; i <= epochs; i++)
                {
                    var cost = UpdateParameters(data, label, lr);
                    if (i % recordInterval == 0)
                    {
                        costs.Add(cost[0, 0]);
                    }
                }
                return costs;
            }
        }
    }
}
