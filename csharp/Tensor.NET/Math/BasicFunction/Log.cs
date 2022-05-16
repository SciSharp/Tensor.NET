using Tensornet.Common;

namespace Tensornet.Math{
    public static partial class MathT{
        public static Tensor<double> Log(Tensor<double> inp, double y){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Log(y, x));
        }
        public static Tensor<float> Log(Tensor<float> inp, float y){
            return OnElemOperation.Execute<float, float>(inp, x => System.MathF.Log(y, x));
        }
        public static Tensor<double> Log(Tensor<long> inp, double y){
            return OnElemOperation.Execute<long, double>(inp, x => System.Math.Log(y, x));
        }
        public static Tensor<double> Log(Tensor<int> inp, double y){
            return OnElemOperation.Execute<int, double>(inp, x => System.Math.Log(y, x));
        }
        public static Tensor<double> Log(double baseValue, Tensor<double> inp){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Log(x, baseValue));
        }
        public static Tensor<float> Log(float baseValue, Tensor<float> inp){
            return OnElemOperation.Execute<float, float>(inp, x => System.MathF.Log(x, baseValue));
        }
        public static Tensor<double> Log(double baseValue, Tensor<long> inp){
            return OnElemOperation.Execute<long, double>(inp, x => System.Math.Log(x, baseValue));
        }
        public static Tensor<double> Log(double baseValue, Tensor<int> inp){
            return OnElemOperation.Execute<int, double>(inp, x => System.Math.Log(x, baseValue));
        }
    }
}