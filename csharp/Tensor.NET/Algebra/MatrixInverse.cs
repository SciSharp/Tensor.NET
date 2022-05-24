using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;

namespace Tensornet{
    public static class InverseExtension{
        /// <summary>
        /// Get the inverse of the tensor. For the details of the manipulation, please refer to https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        public static Tensor<double> Inverse(this Tensor<int> src)
        {
            Tensor<double> res = new Tensor<double>(DeduceLayout(src.TLayout));
            res.TLayout.InitContiguousLayout();
            InverseInternal<double>(src.ToTensor<double>(), res);
            return res;
        }
        /// <summary>
        /// Get the inverse of the tensor. For the details of the manipulation, please refer to https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        public static Tensor<double> Inverse(this Tensor<long> src)
        {
            Tensor<double> res = new Tensor<double>(DeduceLayout(src.TLayout));
            res.TLayout.InitContiguousLayout();
            InverseInternal<double>(src.ToTensor<double>(), res);
            return res;
        }
        /// <summary>
        /// Get the inverse of the tensor. For the details of the manipulation, please refer to https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        public static Tensor<double> Inverse(this Tensor<double> src)
        {
            Tensor<double> res = new Tensor<double>(DeduceLayout(src.TLayout));
            res.TLayout.InitContiguousLayout();
            if(src.TLayout.IsContiguous()){
                InverseInternal<double>(src, res);
            }
            else{
                InverseInternal<double>(src.ToContiguousTensor(), res);
            }
            return res;
        }
        /// <summary>
        /// Get the inverse of the tensor. For the details of the manipulation, please refer to https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        public static Tensor<float> Inverse(this Tensor<float> src)
        {
            Tensor<float> res = new Tensor<float>(DeduceLayout(src.TLayout));
            res.TLayout.InitContiguousLayout();
            if(src.TLayout.IsContiguous()){
                InverseInternal<float>(src, res);
            }
            else{
                InverseInternal<float>(src.ToContiguousTensor(), res);
            }
            return res;
        }
        private unsafe static void InverseInternal<T>(Tensor<T> src, Tensor<T> dst) where T : struct, IEquatable<T>, IConvertible{
            IntPtr status = NativeExecutor.Execute(NativeApi.MatrixInverse, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, IntPtr.Zero, Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
        private static TensorLayout DeduceLayout(TensorLayout src){
            TensorLayout res = new TensorLayout();
            if (src.NDim < 2) {
                throw new MismatchedShapeException("The tensor to calculate inverse must has at least two dims.");
            }
            if (src.Shape[src.NDim - 1] != src.Shape[src.NDim - 2]) {
                throw new MismatchedShapeException("The tensor to calculate inverse must has its last two dims square.");
            }
            res.DType = (src.DType is DType.Float32 or DType.Float64) ? src.DType : DType.Float64;
            res.NDim = src.NDim;
            for (int i = 0; i < src.NDim; i++) {
                res.Shape[i] = src.Shape[i];
            }
            return res;
        }
    }
}