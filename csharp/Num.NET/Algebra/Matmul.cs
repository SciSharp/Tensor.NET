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
                res.Shape[0] = rhs.Shape[0];
                res.Shape[1] = lhs.Shape[0];
                res.NDim = 2;
                lhs.BroadcastInplace(new TensorShape(lhs.Shape[0], 1));
                rhs.BroadcastInplace(new TensorShape(1, rhs.Shape[0]));
                return res;
            }
            int dim = lhs.NDim > rhs.NDim ? lhs.NDim : rhs.NDim;
            res.NDim = dim;
            List<int> leftDstShape = new List<int>();
            List<int> rightDstShape = new List<int>();
            for (int i = dim - 1; i >= 2; i--) {
                if (i >= lhs.NDim || i < rhs.NDim && lhs.Shape[i] == 1) {
                    res.Shape[i] = rhs.Shape[i];
                } else if (i >= rhs.NDim || i < lhs.NDim && rhs.Shape[i] == 1) {
                    res.Shape[i] = lhs.Shape[i];
                } else if (lhs.Shape[i] == rhs.Shape[i]) {
                    res.Shape[i] = lhs.Shape[i];
                } else {
                    throw new MismatchedShapeException(lhs, rhs, "Matmul");
                }
                leftDstShape.Add(res.Shape[i]);
                rightDstShape.Add(res.Shape[i]);
            }
            if (lhs.NDim == 1 && rhs.Shape[1] != lhs.Shape[0] ||
                rhs.NDim == 1 && rhs.Shape[0] != lhs.Shape[0] ||
                lhs.NDim != 1 && rhs.NDim != 1 && lhs.Shape[0] != rhs.Shape[1]) {
                throw new MismatchedShapeException(lhs, rhs, "Matmul");
            }
            res.Shape[1] = lhs.NDim == 1 ? 1 : lhs.Shape[1];
            res.Shape[0] = rhs.NDim == 1 ? 1 : rhs.Shape[0];
            leftDstShape.Add(lhs.NDim > 1 ? lhs.Shape[1] : 1);
            rightDstShape.Add(rhs.NDim > 1 ? rhs.Shape[1] : rhs.Shape[0]);
            leftDstShape.Add(lhs.Shape[0]);
            rightDstShape.Add(rhs.NDim > 1 ? rhs.Shape[0] : 1);
            if (rhs.NDim == 1) {
                rhs.Shape[1] = rhs.Shape[0];
                rhs.Shape[0] = 1;
                rhs.NDim = 2;
                rhs.Stride[1] = 1;
                rhs.Stride[0] = 0;
            }
            lhs.BroadcastInplace(new TensorShape(leftDstShape));
            rhs.BroadcastInplace(new TensorShape(rightDstShape));
            return res;
        }
    }
}