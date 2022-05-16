using Tensornet.Common;

namespace Tensornet.Math{
    public static class CeilExtension{
        public static Tensor<double> Ceil(this Tensor<double> inp){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Ceiling(x));
        }
        public static Tensor<float> Ceil(this Tensor<float> inp){
            return OnElemOperation.Execute<float, float>(inp, x => System.MathF.Ceiling(x));
        }
    }
}