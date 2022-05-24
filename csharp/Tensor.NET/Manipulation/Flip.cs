using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;

namespace Tensornet{
    public static class FlipExtension{
        /// <summary>
        /// Reverse the order of elements in an array along the given axis. The shape of the array is preserved, but the elements are reordered.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.flip.html?highlight=flip#numpy.flip
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="axes"> Axes along which to flip over. If no axis is given, then the tensor will be flipped on all axes.</param>
        /// <returns></returns>
        /// <exception cref="InvalidParamException"></exception>
        public static Tensor<T> Flip<T>(this Tensor<T> src, params int[] axes) where T : struct, IEquatable<T>, IConvertible
        {
            bool[] dims = new bool[TensorLayout.MAX_NDIM];
            if(axes.Length == 0){
                for (int i = 0; i < src.TLayout.NDim; i++){
                    dims[i] = true;
                }
            }
            else{
                for (int i = 0; i < dims.Length; i++){
                    dims[i] = false;
                }
                foreach (var i in axes)
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
        /// Reverse the order of elements in an array along the given axis. The shape of the array is preserved, but the elements are reordered.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.flip.html?highlight=flip#numpy.flip
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src">the tensor to be flipped</param>
        /// <param name="axes"> The axes to flip. If it's set to null, then all axis will be flipped. </param>
        /// <returns>The flipped tensor</returns>
        public static Tensor<T> Flip<T>(Tensor<T> src, int[]? axes = null) where T : struct, IEquatable<T>, IConvertible{
            if(axes is null){
                return src.Flip();
            }
            else{
                return src.Flip(axes);
            }
        }
        /// <summary>
        /// Reverse the order of elements in an array along the given axis. The shape of the array is preserved, but the elements are reordered.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.flip.html?highlight=flip#numpy.flip
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src">the tensor to be flipped</param>
        /// <param name="axis"> The axis to flip. </param>
        /// <returns>The flipped tensor</returns>
        public static Tensor<T> Flip<T>(Tensor<T> src, int axis) where T : struct, IEquatable<T>, IConvertible{
            return src.Flip(axis);
        }
    }
}