using BenchmarkDotNet.Attributes;
using static Tensorflow.Binding;
using Tensorflow.NumPy;
using Microsoft.Diagnostics.Tracing.Parsers.Clr;

public class TFNetRegressionTest
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
    public void TrainTFNet()
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
            NDArray data = np.random.normal(0, 2.0f, new int[] { BenchmarkParam.FeatureCount, BenchmarkParam.DataLength }).astype(Tensorflow.TF_DataType.TF_DOUBLE);
            NDArray label = np.random.normal(0, 1.0f, new int[] { 1, BenchmarkParam.DataLength }).astype(Tensorflow.TF_DataType.TF_DOUBLE);
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
            w = np.random.normal(0, 0.01f, new int[] { _dataDim, 1 }).astype(Tensorflow.TF_DataType.TF_DOUBLE);
            b = 0;
            return (w, b);
        }
        public (NDArray, NDArray, NDArray) ForwardAndBackwardPropagate(NDArray data, NDArray label)
        {
            var dataNum = data.shape[0];
            // forward propagation
            var z = tf.matmul(tf.transpose(w), data) + b;
            var predict = Sigmoid.Run(z.numpy());
            var diff = predict - label;

            var cost = np.mean(-(np.multiply(label, np.log(predict)) + np.multiply((1 - label), np.log(1 - predict))));

            // back propagation
            var dw = tf.matmul(data, tf.transpose(diff)) / dataNum;
            var db = np.sum(diff) / dataNum;
            return (cost, dw.numpy(), db);
        }

        private NDArray Transpose(NDArray src)
        {
            NDArray res = np.zeros(new long[] { src.shape[1], src.shape[0] });
            for (int i = 0; i < src.shape[0]; i++)
            {
                for (int j = 0; j < src.shape[1]; j++)
                {
                    res[j, i] = src[i, j];
                }
            }
            return res;
        }

        public NDArray UpdataParameters(NDArray data, NDArray label, double lr)
        {
            var (cost, gradW, gradB) = ForwardAndBackwardPropagate(data, label);

            w -= lr * gradW;
            b -= lr * gradB[0];

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
                    costs.Add(cost[0]);
                }
            }
            return costs;
        }
    }
}