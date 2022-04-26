using Numnet.Common;
using Numnet.Native;
using Numnet.Exceptions;

namespace Numnet.Algebra{
    public static class MatmulExtension{
        public static Tensor<T> Matmul<T>(this Tensor<T> lhs, Tensor<T> rhs) where T : struct{
            return MatmulInternal<T, T, T>(lhs, rhs);
        }
        public static Tensor<double> Matmul(this Tensor<double> lhs, Tensor<int> rhs){
            return MatmulInternal<double, int, double>(lhs, rhs);
        }
        public static Tensor<double> Matmul(this Tensor<double> lhs, Tensor<float> rhs){
            return MatmulInternal<double, float, double>(lhs, rhs);
        }
        public static Tensor<double> Matmul(this Tensor<double> lhs, Tensor<long> rhs){
            return MatmulInternal<double, long, double>(lhs, rhs);
        }
        public static Tensor<double> Matmul(this Tensor<double> lhs, Tensor<bool> rhs){
            return MatmulInternal<double, bool, double>(lhs, rhs);
        }
        public static Tensor<double> Matmul(this Tensor<int> lhs, Tensor<double> rhs){
            return MatmulInternal<int, double, double>(lhs, rhs);
        }
        public static Tensor<float> Matmul(this Tensor<int> lhs, Tensor<float> rhs){
            return MatmulInternal<int, float, float>(lhs, rhs);
        }
        public static Tensor<long> Matmul(this Tensor<int> lhs, Tensor<long> rhs){
            return MatmulInternal<int, long, long>(lhs, rhs);
        }
        public static Tensor<int> Matmul(this Tensor<int> lhs, Tensor<bool> rhs){
            return MatmulInternal<int, bool, int>(lhs, rhs);
        }
        public static Tensor<double> Matmul(this Tensor<long> lhs, Tensor<double> rhs){
            return MatmulInternal<long, double, double>(lhs, rhs);
        }
        public static Tensor<float> Matmul(this Tensor<long> lhs, Tensor<float> rhs){
            return MatmulInternal<long, float, float>(lhs, rhs);
        }
        public static Tensor<long> Matmul(this Tensor<long> lhs, Tensor<int> rhs){
            return MatmulInternal<long, int, long>(lhs, rhs);
        }
        public static Tensor<long> Matmul(this Tensor<long> lhs, Tensor<bool> rhs){
            return MatmulInternal<long, bool, long>(lhs, rhs);
        }
        public static Tensor<float> Matmul(this Tensor<float> lhs, Tensor<long> rhs){
            return MatmulInternal<float, long, float>(lhs, rhs);
        }
        public static Tensor<float> Matmul(this Tensor<float> lhs, Tensor<int> rhs){
            return MatmulInternal<float, int, float>(lhs, rhs);
        }
        public static Tensor<float> Matmul(this Tensor<float> lhs, Tensor<bool> rhs){
            return MatmulInternal<float, bool, float>(lhs, rhs);
        }
        public static Tensor<double> Matmul(this Tensor<float> lhs, Tensor<double> rhs){
            return MatmulInternal<float, double, double>(lhs, rhs);
        }
        public static Tensor<double> Matmul(this Tensor<bool> lhs, Tensor<double> rhs){
            return MatmulInternal<bool, double, double>(lhs, rhs);
        }
        public static Tensor<float> Matmul(this Tensor<bool> lhs, Tensor<float> rhs){
            return MatmulInternal<bool, float, float>(lhs, rhs);
        }
        public static Tensor<long> Matmul(this Tensor<bool> lhs, Tensor<long> rhs){
            return MatmulInternal<bool, long, long>(lhs, rhs);
        }
        public static Tensor<int> Matmul(this Tensor<bool> lhs, Tensor<int> rhs){
            return MatmulInternal<bool, int, int>(lhs, rhs);
        }
        private static Tensor<TC> MatmulInternal<TA, TB, TC>(Tensor<TA> lhs, Tensor<TB> rhs) where TA : struct where TB : struct where TC : struct
        {
            TensorLayout leftLayout = new TensorLayout(lhs.TLayout);
            TensorLayout rightLayout = new TensorLayout(rhs.TLayout);
            Tensor<TC> res = new Tensor<TC>(DeduceLayout(leftLayout, rightLayout));
            res.TLayout.InitContiguousLayout();
            IntPtr status = NativeExecutor.Execute(NativeApi.Matmul, lhs.TMemory, rhs.TMemory, res.TMemory, leftLayout, rightLayout, res.TLayout, IntPtr.Zero, Tensor<TC>.Provider);
            NativeStatus.AssertOK(status);
            return res;
        }
        private static TensorLayout DeduceLayout(TensorLayout lhs, TensorLayout rhs){
            TensorLayout res = new TensorLayout();
            res.DType = DefaultTypeDeduce.Deduce(lhs.DType, rhs.DType);
            if (lhs.NDim <= 0 || rhs.NDim <= 0)
                throw new MismatchedShapeException(lhs, rhs, "Matmul");
            if (lhs.IsScalar() && rhs.IsScalar()) {
                res.Shape[0] = 1;
                res.NDim = 1;
                return res;
            }
            if (lhs.NDim == 1 && rhs.NDim == 1) {
                res.Shape[0] = lhs.Shape[0];
                res.Shape[1] = rhs.Shape[0];
                res.NDim = 2;
                lhs.BroadcastInplace(new TensorShape(lhs.Shape[0], 1));
                rhs.BroadcastInplace(new TensorShape(1, rhs.Shape[0]));
                return res;
            }
            int dim = lhs.NDim > rhs.NDim ? lhs.NDim : rhs.NDim;
            int min_dim = lhs.NDim < rhs.NDim ? lhs.NDim : rhs.NDim;
            res.NDim = dim;
            TensorShape aDstShape = new TensorShape(lhs);
            TensorShape bDstShape = new TensorShape(rhs);

            for (int i = 0; i < dim - 2; i++) {
                int aIdx = lhs.NDim - i - 3;
                int bIdx = rhs.NDim - i - 3;
                int targetIdx = dim - i - 3;
                if (aIdx >= 0 &&
                    (bIdx < 0 || lhs.Shape[aIdx] == 1 || rhs.Shape[bIdx] == 1 ||
                    lhs.Shape[aIdx] == rhs.Shape[bIdx])) {
                    if (bIdx < 0)
                        res.Shape[targetIdx] = lhs.Shape[aIdx];
                    else
                        res.Shape[targetIdx] =
                            lhs.Shape[aIdx] == 1 ? System.Math.Max(rhs.Shape[bIdx], lhs.Shape[aIdx]) : lhs.Shape[aIdx];
                } else if (aIdx < 0) {
                    res.Shape[targetIdx] = rhs.Shape[bIdx];
                } else {
                    throw new MismatchedShapeException(lhs, rhs, "Matmul");
                }
                aDstShape.Shape[targetIdx] = res.Shape[targetIdx];
                bDstShape.Shape[targetIdx] = res.Shape[targetIdx];
            }
            if (lhs.NDim == 1 && rhs.Shape[rhs.NDim - 2] != lhs.Shape[0] ||
                rhs.NDim == 1 && rhs.Shape[0] != lhs.Shape[lhs.NDim - 1] ||
                lhs.NDim != 1 && rhs.NDim != 1 &&
                    lhs.Shape[lhs.NDim - 1] != rhs.Shape[rhs.NDim - 2]) {
                throw new MismatchedShapeException(lhs, rhs, "Matmul");
            }
            res.Shape[dim - 2] = aDstShape.Shape[dim - 2] =
                lhs.NDim == 1 ? 1 : lhs.Shape[lhs.NDim - 2];
            res.Shape[dim - 1] = bDstShape.Shape[dim - 1] =
                rhs.NDim == 1 ? 1 : rhs.Shape[rhs.NDim - 1];
            aDstShape.Shape[dim - 1] = bDstShape.Shape[dim - 2] = lhs.Shape[lhs.NDim - 1];
            if (rhs.NDim == 1) {
                rhs.Shape[1] = 1;
                rhs.NDim = 2;
                rhs.Stride[0] = 1;
                rhs.Stride[1] = 0;
            }
            aDstShape.NDim = bDstShape.NDim = res.NDim;
            lhs.BroadcastInplace(aDstShape);
            rhs.BroadcastInplace(bDstShape);
            return res;
        }
    }
}