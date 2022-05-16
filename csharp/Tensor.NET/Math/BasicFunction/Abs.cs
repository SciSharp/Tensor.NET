using Tensornet.Common;

namespace Tensornet.Math{
    public static class AbsExtension{
        public static Tensor<double> Add(this Tensor<double> inp){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Abs(x));
        }
        public static Tensor<float> Add(this Tensor<float> inp){
            return OnElemOperation.Execute<float, float>(inp, x => System.Math.Abs(x));
        }
        public static Tensor<long> Add(this Tensor<long> inp){
            return OnElemOperation.Execute<long, long>(inp, x => System.Math.Abs(x));
        }
        public static Tensor<int> Add(this Tensor<int> inp){
            return OnElemOperation.Execute<int, int>(inp, x => System.Math.Abs(x));
        }
        public static Tensor<bool> Add(this Tensor<bool> inp){
            return OnElemOperation.Execute<bool, bool>(inp, x => x);
        }
    }
}