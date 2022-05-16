using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Exceptions;

namespace Tensornet{
    public static class DotExtension{
        public static Tensor<T> Dot<T>(this Tensor<T> lhs, Tensor<T> rhs) where T : struct, IEquatable<T>, IConvertible{
            return DotInternal<T, T, T>(lhs, rhs);
        }
        public static Tensor<double> Dot(this Tensor<double> lhs, Tensor<int> rhs){
            return DotInternal<double, int, double>(lhs, rhs);
        }
        public static Tensor<double> Dot(this Tensor<double> lhs, Tensor<float> rhs){
            return DotInternal<double, float, double>(lhs, rhs);
        }
        public static Tensor<double> Dot(this Tensor<double> lhs, Tensor<long> rhs){
            return DotInternal<double, long, double>(lhs, rhs);
        }
        public static Tensor<double> Dot(this Tensor<double> lhs, Tensor<bool> rhs){
            return DotInternal<double, bool, double>(lhs, rhs);
        }
        public static Tensor<double> Dot(this Tensor<int> lhs, Tensor<double> rhs){
            return DotInternal<int, double, double>(lhs, rhs);
        }
        public static Tensor<float> Dot(this Tensor<int> lhs, Tensor<float> rhs){
            return DotInternal<int, float, float>(lhs, rhs);
        }
        public static Tensor<long> Dot(this Tensor<int> lhs, Tensor<long> rhs){
            return DotInternal<int, long, long>(lhs, rhs);
        }
        public static Tensor<int> Dot(this Tensor<int> lhs, Tensor<bool> rhs){
            return DotInternal<int, bool, int>(lhs, rhs);
        }
        public static Tensor<double> Dot(this Tensor<long> lhs, Tensor<double> rhs){
            return DotInternal<long, double, double>(lhs, rhs);
        }
        public static Tensor<float> Dot(this Tensor<long> lhs, Tensor<float> rhs){
            return DotInternal<long, float, float>(lhs, rhs);
        }
        public static Tensor<long> Dot(this Tensor<long> lhs, Tensor<int> rhs){
            return DotInternal<long, int, long>(lhs, rhs);
        }
        public static Tensor<long> Dot(this Tensor<long> lhs, Tensor<bool> rhs){
            return DotInternal<long, bool, long>(lhs, rhs);
        }
        public static Tensor<float> Dot(this Tensor<float> lhs, Tensor<long> rhs){
            return DotInternal<float, long, float>(lhs, rhs);
        }
        public static Tensor<float> Dot(this Tensor<float> lhs, Tensor<int> rhs){
            return DotInternal<float, int, float>(lhs, rhs);
        }
        public static Tensor<float> Dot(this Tensor<float> lhs, Tensor<bool> rhs){
            return DotInternal<float, bool, float>(lhs, rhs);
        }
        public static Tensor<double> Dot(this Tensor<float> lhs, Tensor<double> rhs){
            return DotInternal<float, double, double>(lhs, rhs);
        }
        public static Tensor<double> Dot(this Tensor<bool> lhs, Tensor<double> rhs){
            return DotInternal<bool, double, double>(lhs, rhs);
        }
        public static Tensor<float> Dot(this Tensor<bool> lhs, Tensor<float> rhs){
            return DotInternal<bool, float, float>(lhs, rhs);
        }
        public static Tensor<long> Dot(this Tensor<bool> lhs, Tensor<long> rhs){
            return DotInternal<bool, long, long>(lhs, rhs);
        }
        public static Tensor<int> Dot(this Tensor<bool> lhs, Tensor<int> rhs){
            return DotInternal<bool, int, int>(lhs, rhs);
        }
        private static Tensor<TC> DotInternal<TA, TB, TC>(Tensor<TA> lhs, Tensor<TB> rhs) 
        where TA : struct, IEquatable<TA>, IConvertible
        where TB : struct, IEquatable<TB>, IConvertible
        where TC : struct, IEquatable<TC>, IConvertible
        {
            TensorLayout leftLayout = new TensorLayout(lhs.TLayout);
            TensorLayout rightLayout = new TensorLayout(rhs.TLayout);
            Tensor<TC> res = new Tensor<TC>(DeduceLayout(leftLayout, rightLayout));
            res.TLayout.InitContiguousLayout();
            IntPtr status = NativeExecutor.Execute(NativeApi.Dot, lhs.TMemory, rhs.TMemory, res.TMemory, leftLayout, rightLayout, res.TLayout, IntPtr.Zero, Tensor<TC>.Provider);
            NativeStatus.AssertOK(status);
            return res;
        }
        private static TensorLayout DeduceLayout(TensorLayout lhs, TensorLayout rhs){
            TensorLayout res = new TensorLayout();
            res.DType = DefaultTypeDeduce.Deduce(lhs.DType, rhs.DType);
            if (lhs.NDim <= 0 || rhs.NDim <= 0)
                throw new MismatchedShapeException(lhs, rhs, "Dot");
            int dim = System.Math.Max(lhs.NDim, rhs.NDim);
            for (int i = 0, j = 0, k = 0; i < lhs.NDim && j < rhs.NDim; i++, j++, k++) {
                int a_idx = lhs.NDim - i - 1;
                int b_idx = lhs.NDim - j - 1;
                if (lhs.Shape[a_idx] != rhs.Shape[b_idx] && lhs.Shape[a_idx] != 1 &&
                    rhs.Shape[b_idx] != 1) {
                    throw new MismatchedShapeException("Cannot broadcast bool index to the shape of target tensor.");
                } else if (lhs.Shape[a_idx] == rhs.Shape[b_idx]) {
                    res.Shape[dim - k - 1] = lhs.Shape[a_idx];
                } else if (lhs.Shape[a_idx] == 1) {
                    res.Shape[dim - k - 1] = lhs.Shape[b_idx];
                } else if (rhs.Shape[a_idx] == 1) {
                    res.Shape[dim - k - 1] = lhs.Shape[a_idx];
                } else {
                    throw new MismatchedShapeException("Unknown error when deducing the layout.");
                }
            }
            for (int i = System.Math.Min(lhs.NDim, rhs.NDim); i < lhs.NDim; i++) {
                res.Shape[dim - i - 1] = lhs.Shape[lhs.NDim - i - 1];
            }
            for (int i = System.Math.Min(lhs.NDim, rhs.NDim); i < rhs.NDim; i++) {
                res.Shape[dim - i - 1] = rhs.Shape[rhs.NDim - i - 1];
            }
            res.NDim = dim;
            lhs.BroadcastInplace(res);
            rhs.BroadcastInplace(res);
            return res;
        }
    }
}