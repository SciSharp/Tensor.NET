using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;

namespace Tensornet{
    public static class TransposeExtension{

        public static Tensor<T> Transpose<T>(this Tensor<T> src, int dimA, int dimB) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout, dimA, dimB));
            res.TLayout.InitContiguousLayout();
            TransposeInternal(src, res, dimA, dimB);
            return res;
        }
        private unsafe static void TransposeInternal<T>(Tensor<T> src, Tensor<T> dst, int dimA, int dimB) where T : struct, IEquatable<T>, IConvertible{
            TransposeParam param = new TransposeParam() { dimA = dimA, dimB = dimB };
            IntPtr status = NativeExecutor.Execute(NativeApi.Transpose, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
        private static TensorLayout DeduceLayout(TensorLayout src, int dimA, int dimB){
            TensorLayout res = new TensorLayout();
            if (dimA >= src.NDim || dimB >= src.NDim) {
                throw new InvalidParamException("Invalid param for transpose.");
            }
            res.DType = src.DType;
            res.NDim = src.NDim;
            for (int i = 0; i < src.NDim; i++) {
                res.Shape[i] = src.Shape[i];
            }
            res.Shape[dimA] = src.Shape[dimB];
            res.Shape[dimB] = src.Shape[dimA];
            return res;
        }
    }
}