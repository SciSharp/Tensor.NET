using Numnet.Common;
using Numnet.Native;
using Numnet.Exceptions;

namespace Numnet.Algebra{
    public static class MatmulExtension{

        public static Tensor Matmul(this Tensor lhs, Tensor rhs)
        {
            TensorLayout leftLayout = new TensorLayout(lhs.TLayout);
            TensorLayout rightLayout = new TensorLayout(rhs.TLayout);
            Tensor res = new Tensor(DeduceLayout(leftLayout, rightLayout));
            res.TLayout.InitContiguousLayout();
            IntPtr status = Tensor.Execute(NativeApi.Matmul, lhs.TMemory, rhs.TMemory, res.TMemory, leftLayout, rightLayout, res.TLayout, IntPtr.Zero, Tensor.Provider);
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
                            lhs.Shape[aIdx] == 1 ? Math.Max(rhs.Shape[bIdx], lhs.Shape[aIdx]) : lhs.Shape[aIdx];
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