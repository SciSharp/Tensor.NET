using Numnet.Common;

namespace Numnet.Math{
    public static partial class MathT{
        public static Tensor<double> Clamp(Tensor<double> inp, double min, double max){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Clamp(x, min, max));
        }
        public static Tensor<float> Clamp(Tensor<float> inp, float min, float max){
            return OnElemOperation.Execute<float, float>(inp, x => System.Math.Clamp(x, min, max));
        }
        public static Tensor<long> Clamp(Tensor<long> inp, long min, long max){
            return OnElemOperation.Execute<long, long>(inp, x => System.Math.Clamp(x, min, max));
        }
        public static Tensor<int> Clamp(Tensor<int> inp, int min, int max){
            return OnElemOperation.Execute<int, int>(inp, x => System.Math.Clamp(x, min, max));
        }
    }
}