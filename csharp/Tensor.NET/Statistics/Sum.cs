using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;

namespace Tensornet{
    public static class SumExtension{
        public static Tensor<T> Sum<T>(this Tensor<T> src, int[] axes, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout, axes));
            res.TLayout.InitContiguousLayout();
            bool[] boolDims = new bool[src.TLayout.NDim];
            var span = boolDims.AsSpan();
            span.Fill(false);
            foreach(var axis in axes){
                span[axis] = true;
            }
            SumInternal(src, res, boolDims, keepDims);
            return res;
        }
        public static Tensor<T> Sum<T>(this Tensor<T> src, int axis, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout, axis));
            res.TLayout.InitContiguousLayout();
            bool[] boolDims = new bool[src.TLayout.NDim];
            var span = boolDims.AsSpan();
            span.Fill(false);
            span[axis] = true;
            SumInternal(src, res, boolDims, keepDims);
            return res;
        }
        public static Tensor<T> Sum<T>(this Tensor<T> src, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout));
            res.TLayout.InitContiguousLayout();
            bool[] boolDims = new bool[src.TLayout.NDim];
            boolDims.AsSpan().Fill(true);
            SumInternal(src, res, boolDims, keepDims);
            return res;
        }
        private unsafe static void SumInternal<T>(Tensor<T> src, Tensor<T> dst, bool[] dims, bool keepDims) where T : struct, IEquatable<T>, IConvertible{
            fixed(bool* ptr = dims){
                SumParam p = new SumParam() { dims = new IntPtr(ptr) };
                IntPtr status = NativeExecutor.Execute(NativeApi.Sum, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&p), Tensor<T>.Provider);
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
        /// <summary>
        /// Sum the tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The tensor to be sumed. </param>
        /// <param name="axes"> The axes to sum. </param>
        /// <param name="keepDims"> Whether to keep the dims after the sum. If false, the NDim of the result may be different with the input. </param>
        /// <returns>The Sumped tensor</returns>
        public static Tensor<T> Sum<T>(Tensor<T> src, int[] axes, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible{
            return src.Sum(axes, keepDims);
        }
        /// <summary>
        /// Sum the tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The tensor to be sumed. </param>
        /// <param name="axis"> The axis to sum. </param>
        /// <param name="keepDims"> Whether to keep the dims after the sum. If false, the NDim of the result may be different with the input. </param>
        /// <returns>The Sumped tensor</returns>
        public static Tensor<T> Sum<T>(Tensor<T> src, int axis, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible{
            return src.Sum(axis, keepDims);
        }
        /// <summary>
        /// Sum the tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The tensor to be sumed. </param>
        /// <param name="keepDims"> Whether to keep the dims after the sum. If false, the NDim of the result may be different with the input. </param>
        /// <returns>The Sumped tensor</returns>
        public static Tensor<T> Sum<T>(Tensor<T> src, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible{
            return src.Sum(keepDims);
        }
    }
}