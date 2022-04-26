using Numnet.Common;

namespace Numnet.Math{
    public static class FloorExtension{
        public static Tensor<double> Floor(this Tensor<double> inp){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Floor(x));
        }
        public static Tensor<float> Floor(this Tensor<float> inp){
            return OnElemOperation.Execute<float, float>(inp, x => System.MathF.Floor(x));
        }

    }
}