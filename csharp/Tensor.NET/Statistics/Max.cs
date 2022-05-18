using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;

namespace Tensornet{
    public static class MaxExtension{
        public static Tensor<T> Max<T>(this Tensor<T> src, int[] axes, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout, axes));
            res.TLayout.InitContiguousLayout();
            bool[] boolDims = new bool[src.TLayout.NDim];
            var span = boolDims.AsSpan();
            span.Fill(false);
            foreach(var axis in axes){
                span[axis] = true;
            }
            MaxInternal(src, res, boolDims, keepDims);
            return res;
        }
        public static Tensor<T> Max<T>(this Tensor<T> src, int axis, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout, axis));
            res.TLayout.InitContiguousLayout();
            bool[] boolDims = new bool[src.TLayout.NDim];
            var span = boolDims.AsSpan();
            span.Fill(false);
            span[axis] = true;
            MaxInternal(src, res, boolDims, keepDims);
            return res;
        }
        public static Tensor<T> Max<T>(this Tensor<T> src, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout));
            res.TLayout.InitContiguousLayout();
            bool[] boolDims = new bool[src.TLayout.NDim];
            boolDims.AsSpan().Fill(true);
            MaxInternal(src, res, boolDims, keepDims);
            return res;
        }
        private unsafe static void MaxInternal<T>(Tensor<T> src, Tensor<T> dst, bool[] dims, bool keepDims) where T : struct, IEquatable<T>, IConvertible{
            fixed(bool* ptr = dims){
                MaxParam p = new MaxParam() { dims = new IntPtr(ptr) };
                IntPtr status = NativeExecutor.Execute(NativeApi.Max, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&p), Tensor<T>.Provider);
                NativeStatus.AssertOK(status);
            }
            if(!keepDims){
                dst.TLayout.RemoveAllDanglingAxisInplace();
            }
        }
        private static TensorLayout DeduceLayout(TensorLayout src, int[] axes){
            var res = new TensorLayout(src, true);
            foreach(var dim in axes){
                res.Shape[dim] = 1;
            }
            return res;
        }
        private static TensorLayout DeduceLayout(TensorLayout src, int axis){
            var res = new TensorLayout(src, true);
            res.Shape[axis] = 1;
            return res;
        }
        private static TensorLayout DeduceLayout(TensorLayout src){
            var res = new TensorLayout(src, true);
            res.Shape.AsSpan().Fill(1);
            return res;
        }
    }

    public static partial class Tensor{
        /// <Summary>
        /// Get the maximum elements of the tensor.
        /// </Summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The tensor to get maximum elements. </param>
        /// <param name="axes"> The axes to execute. </param>
        /// <param name="keepDims"> Whether to keep the dims after the Max. If false, the NDim of the result may be different with the input. </param>
        /// <returns></returns>
        public static Tensor<T> Max<T>(Tensor<T> src, int[] axes, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible{
            return src.Max(axes, keepDims);
        }
        /// <Summary>
        /// Get the maximum elements of the tensor.
        /// </Summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The tensor to get maximum elements. </param>
        /// <param name="axis"> The axis to execute. </param>
        /// <param name="keepDims"> Whether to keep the dims after the Max. If false, the NDim of the result may be different with the input. </param>
        /// <returns></returns>
        public static Tensor<T> Max<T>(Tensor<T> src, int axis, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible{
            return src.Max(axis, keepDims);
        }
        /// <Summary>
        /// Get the maximum elements of the tensor.
        /// </Summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The tensor to get maximum elements. </param>
        /// <param name="keepDims"> Whether to keep the dims after the Max. If false, the NDim of the result may be different with the input. </param>
        /// <returns></returns>
        public static Tensor<T> Max<T>(Tensor<T> src, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible{
            return src.Max(keepDims);
        }
    }
}