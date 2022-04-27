using Numnet.Common;

namespace Numnet.Math{
    public static partial class MathT{
        public static Tensor<double> Exp(Tensor<double> inp){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Exp(x));
        }
        public static Tensor<float> Exp(Tensor<float> inp){
            return OnElemOperation.Execute<float, float>(inp, x => System.MathF.Exp(x));
        }
        public static Tensor<double> Exp(Tensor<long> inp){
            return OnElemOperation.Execute<long, double>(inp, x => System.Math.Exp(x));
        }
        public static Tensor<double> Exp(Tensor<int> inp){
            return OnElemOperation.Execute<int, double>(inp, x => System.Math.Exp(x));
        }
        public static Tensor<double> Exp(Tensor<bool> inp){
            return OnElemOperation.Execute<bool, double>(inp, x => x ? System.Math.E : 1);
        }
    }
}