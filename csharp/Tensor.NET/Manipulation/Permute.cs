using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;

namespace Tensornet{
    public static class PermuteExtension{
        /// <summary>
        /// Permute the axes of a tensor.
        /// For example, the input is s tensor with shape [2, 3, 4], the parameter is [2, 0, 1], the shapee of the result is [4, 2, 3].
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="dims"> The new arrangement of the dims.</param>
        /// <returns></returns>
        public static Tensor<T> Permute<T>(this Tensor<T> src, params int[] dims) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout, dims));
            res.TLayout.InitContiguousLayout();
            PermuteInternal(src, res, dims);
            return res;
        }
        private unsafe static void PermuteInternal<T>(Tensor<T> src, Tensor<T> dst, int[] dims) where T : struct, IEquatable<T>, IConvertible{
            fixed(int* pdims = dims){
                PermuteParam p = new PermuteParam() { dims = new IntPtr(pdims) };
                IntPtr status = NativeExecutor.Execute(NativeApi.Permute, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&p), Tensor<T>.Provider);
                NativeStatus.AssertOK(status);
            }
        }
        private static TensorLayout DeduceLayout(TensorLayout src, int[] dims){
            TensorLayout res = new TensorLayout();
            int duplicated = 0;
            res.DType = src.DType;
            res.NDim = src.NDim;
            for (int i = 0; i < src.NDim; i++) {
                if (dims[i] >= src.NDim) {
                    throw new InvalidParamException("Invalid param for permute.");
                }
                res.Shape[i] = src.Shape[dims[i]];
                if ((duplicated & (1 << dims[i])) != 0) {
                    throw new InvalidParamException("Duplicated index in the param of permute.");
                }
                duplicated |= 1 << dims[i];
            }
            return res;
        }
    }
}