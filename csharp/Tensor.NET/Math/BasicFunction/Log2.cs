using Tensornet.Common;

namespace Tensornet.Math{
    public static partial class MathT{
        public static Tensor<double> Log2(Tensor<double> inp){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Log2(x));
        }
        public static Tensor<float> Log2(Tensor<float> inp){
            return OnElemOperation.Execute<float, float>(inp, x => System.MathF.Log2(x));
        }
        public static Tensor<double> Log2(Tensor<long> inp){
            return OnElemOperation.Execute<long, double>(inp, x => System.Math.Log2(x));
        }
        public static Tensor<double> Log2(Tensor<int> inp){
            return OnElemOperation.Execute<int, double>(inp, x => System.Math.Log2(x));
        }
    }
}