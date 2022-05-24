using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;

namespace Tensornet{
    public enum SortOrder{
        Increase = 0,
        Decrease = 1
    }
    public static class SortExtension{
        /// <summary>
        /// Return a sorted copy of a tensor.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.sort.html?highlight=sort#numpy.sort
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="axis"> Axis along which to sort. By default it sorts the last axis. Please set it to zero or a positive number.</param>
        /// <param name="order"> The order of the sort. </param>
        /// <returns></returns>
        /// <exception cref="InvalidParamException"></exception>
        public static Tensor<T> Sort<T>(this Tensor<T> src, int axis = -1, SortOrder order = SortOrder.Increase) where T : struct, IEquatable<T>, IConvertible
        {
            if(axis == -1){
                axis = src.TLayout.NDim - 1;
            }
            if(axis <= -2 || axis >= src.TLayout.NDim){
                throw new InvalidParamException("The axis to sort exceeds the range of dims");
            }
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout));
            res.TLayout.InitContiguousLayout();
            SortInternal(src, res, axis, order);
            return res;
        }
        private unsafe static void SortInternal<T>(Tensor<T> src, Tensor<T> dst, int axis, SortOrder order) where T : struct, IEquatable<T>, IConvertible{
            SortParam p = new SortParam() { axis = axis, order = order };
            IntPtr status = NativeExecutor.Execute(NativeApi.Sort, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&p), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
        private static TensorLayout DeduceLayout(TensorLayout src){
            return new TensorLayout(src, true);
        }
    }

    public static partial class Tensor{
        /// <summary>
        /// Return a sorted copy of a tensor.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.sort.html?highlight=sort#numpy.sort
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="axis"> Axis along which to sort. By default it sorts the last axis. Please set it to zero or a positive number.</param>
        /// <param name="order"> The order of the sort. </param>
        /// <returns></returns>
        /// <exception cref="InvalidParamException"></exception>
        public static Tensor<T> Sort<T>(Tensor<T> src, int axis = -1, SortOrder order = SortOrder.Increase) where T : struct, IEquatable<T>, IConvertible{
            return src.Sort(axis, order);
        }
    }
}