using Tensornet.Common;

namespace Tensornet.Math{
    public static partial class MathT{
        public static Tensor<double> Log10(Tensor<double> inp){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Log10(x));
        }
        public static Tensor<float> Log10(Tensor<float> inp){
            return OnElemOperation.Execute<float, float>(inp, x => System.MathF.Log10(x));
        }
        public static Tensor<double> Log10(Tensor<long> inp){
            return OnElemOperation.Execute<long, double>(inp, x => System.Math.Log10(x));
        }
        public static Tensor<double> Log10(Tensor<int> inp){
            return OnElemOperation.Execute<int, double>(inp, x => System.Math.Log10(x));
        }
    }
}