using Numnet.Native;
using Numnet.Exceptions;
using Numnet.Native.Param;

namespace Numnet{
    public enum SortOrder{
        Increase = 0,
        Decrease = 1
    }
    public static class SortExtension{
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
        /// Sort the tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The tensor to be sorted. </param>
        /// <param name="axis"> The axis to Sort. If it's set to -1, then the last axis will be Sorted</param>
        /// <returns>The Sortped tensor</returns>
        public static Tensor<T> Sort<T>(Tensor<T> src, int axis = -1, SortOrder order = SortOrder.Increase) where T : struct, IEquatable<T>, IConvertible{
            return src.Sort(axis, order);
        }
    }
}