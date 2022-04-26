using Numnet.Common;

namespace Numnet.Math{
    public static class RoundExtension{
        public static Tensor<double> Round(this Tensor<double> inp){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Round(x));
        }
        public static Tensor<float> Round(this Tensor<float> inp){
            return OnElemOperation.Execute<float, float>(inp, x => System.MathF.Round(x));
        }
    }
}