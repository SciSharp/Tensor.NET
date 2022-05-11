using Numnet.Common;
using Numnet.Native;
using Numnet.Exceptions;
using Numnet.Native.Param;

namespace Numnet{
    public static class RotateExtension{
        /// <summary>
        /// Rotate the tensor clockwise.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="k"> The times to rotate, which should be in range [1, 3]. The tensor will be rotated by k * 90 degrees. </param>
        /// <param name="dimA"> The first dim of the matrix to rotate. </param>
        /// <param name="dimB"> The second dim of the matrix to rotate. </param>
        /// <returns></returns>
        public static Tensor<T> Rotate<T>(this Tensor<T> src, int k, int dimA, int dimB) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout, k, dimA, dimB));
            res.TLayout.InitContiguousLayout();
            RotateInternal(src, res, k, dimA, dimB);
            return res;
        }
        private unsafe static void RotateInternal<T>(Tensor<T> src, Tensor<T> dst, int k, int dimA, int dimB) where T : struct, IEquatable<T>, IConvertible{
            RotateParam param = new RotateParam(k, dimA, dimB);
            IntPtr status = NativeExecutor.Execute(NativeApi.Rotate, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
        private static TensorLayout DeduceLayout(TensorLayout src, int k, int dimA, int dimB){
            TensorLayout res = new TensorLayout();
            if (k <= 0 || k > 3) {
                throw new InvalidParamException("The k must be between 1 and 3.");
            }
            if (dimA == dimB) {
                throw new InvalidParamException("The dimA and dimB cannot be the same.");
            }
            if (src.NDim <= 1) {
                throw new MismatchedShapeException("The tensor to be rotated must have at least two dims.");
            }
            if (dimA < 0 || dimB < 0 || dimA >= src.NDim ||
                dimB >= src.NDim) {
                throw new InvalidParamException("The dimA or dimB exceeds the valid range.");
            }
            res.DType = src.DType;
            res.NDim = src.NDim;
            for (int i = 0; i < src.NDim; i++) {
                res.Shape[i] = src.Shape[i];
            }
            if (k != 2) {
                (res.Shape[dimA], res.Shape[dimB]) = (res.Shape[dimB], res.Shape[dimA]);
            }
            return res;
        }
    }

    public static partial class Tensor{
        /// <summary>
        /// Rotate the tensor clockwise.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The tensor to be rotated. </param>
        /// <param name="k"> The times to rotate, which should be in range [1, 3]. The tensor will be rotated by k * 90 degrees. </param>
        /// <param name="dimA"> The first dim of the matrix to rotate. </param>
        /// <param name="dimB"> The second dim of the matrix to rotate. </param>
        /// <returns></returns>
        public static Tensor<T> Rotate<T>(Tensor<T> src, int k, int dimA, int dimB) where T : struct, IEquatable<T>, IConvertible{
            return src.Rotate(k, dimA, dimB);
        }
    }
}