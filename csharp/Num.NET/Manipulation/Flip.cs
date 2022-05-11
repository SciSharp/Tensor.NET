using Numnet.Common;
using Numnet.Native;
using Numnet.Exceptions;
using Numnet.Native.Param;

namespace Numnet{
    public static class FlipExtension{

        public static Tensor<T> Flip<T>(this Tensor<T> src, params int[] axis) where T : struct, IEquatable<T>, IConvertible
        {
            bool[] dims = new bool[TensorLayout.MAX_NDIM];
            if(axis.Length == 0){
                for (int i = 0; i < src.TLayout.NDim; i++){
                    dims[i] = true;
                }
            }
            else{
                for (int i = 0; i < dims.Length; i++){
                    dims[i] = false;
                }
                foreach (var i in axis)
                {
                    if (i >= src.TLayout.NDim)
                    {
                        throw new InvalidParamException("The axis to flip exceeds the max dim of the tensor.");
                    }
                    else if (i < 0)
                    {
                        throw new InvalidParamException("The axis to flip must be a positive number");
                    }
                    else
                    {
                        dims[i] = true;
                    }
                }
            }
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout));
            res.TLayout.InitContiguousLayout();
            FlipInternal(src, res, dims);
            return res;
        }
        private unsafe static void FlipInternal<T>(Tensor<T> src, Tensor<T> dst, bool[] dims) where T : struct, IEquatable<T>, IConvertible{
            fixed(bool* pdims = dims){
                FlipParam p = new FlipParam() { dims = new IntPtr(pdims) };
                IntPtr status = NativeExecutor.Execute(NativeApi.Flip, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&p), Tensor<T>.Provider);
                NativeStatus.AssertOK(status);
            }
        }
        private static TensorLayout DeduceLayout(TensorLayout src){
            return new TensorLayout(src, true);
        }
    }

    public static partial class Tensor{
        /// <summary>
        /// Flip the tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src">the tensor to be flipped</param>
        /// <param name="axis"> The axis to flip. If it's set to null, then all axis will be flipped</param>
        /// <returns>The flipped tensor</returns>
        public static Tensor<T> Flip<T>(Tensor<T> src, int[]? axis = null) where T : struct, IEquatable<T>, IConvertible{
            if(axis is null){
                return src.Flip();
            }
            else{
                return src.Flip(axis);
            }
        }
        public static Tensor<T> Flip<T>(Tensor<T> src, int axis) where T : struct, IEquatable<T>, IConvertible{
            return src.Flip(axis);
        }
    }
}