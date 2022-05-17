using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;

namespace Tensornet{
    public static class MeanExtension{
        public static Tensor<double> Mean<T>(this Tensor<T> src, int[] axes, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<double> res = new Tensor<double>(DeduceLayout(src.TLayout, axes));
            res.TLayout.InitContiguousLayout();
            bool[] boolDims = new bool[src.TLayout.NDim];
            var span = boolDims.AsSpan();
            span.Fill(false);
            foreach(var axis in axes){
                span[axis] = true;
            }
            MeanInternal(src, res, boolDims, keepDims);
            return res;
        }
        public static Tensor<double> Mean<T>(this Tensor<T> src, int axis, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<double> res = new Tensor<double>(DeduceLayout(src.TLayout, axis));
            res.TLayout.InitContiguousLayout();
            bool[] boolDims = new bool[src.TLayout.NDim];
            var span = boolDims.AsSpan();
            span.Fill(false);
            span[axis] = true;
            MeanInternal(src, res, boolDims, keepDims);
            return res;
        }
        public static Tensor<double> Mean<T>(this Tensor<T> src, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<double> res = new Tensor<double>(DeduceLayout(src.TLayout));
            res.TLayout.InitContiguousLayout();
            bool[] boolDims = new bool[src.TLayout.NDim];
            boolDims.AsSpan().Fill(true);
            MeanInternal(src, res, boolDims, keepDims);
            return res;
        }
        private unsafe static void MeanInternal<T>(Tensor<T> src, Tensor<double> dst, bool[] dims, bool keepDims) where T : struct, IEquatable<T>, IConvertible{
            fixed(bool* ptr = dims){
                MeanParam p = new MeanParam() { dims = new IntPtr(ptr), keepDims = keepDims };
                IntPtr status = NativeExecutor.Execute(NativeApi.Mean, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&p), Tensor<T>.Provider);
                NativeStatus.AssertOK(status);
            }
            if(!keepDims){
                dst.TLayout.RemoveAllDanglingAxisInplace();
            }
        }
        private static TensorLayout DeduceLayout(TensorLayout src, int[] axes){
            var res = new TensorLayout(src.Shape, DType.Float64);
            foreach(var dim in axes){
                res.Shape[dim] = 1;
            }
            return res;
        }
        private static TensorLayout DeduceLayout(TensorLayout src, int axis){
            var res = new TensorLayout(src.Shape, DType.Float64);
            res.Shape[axis] = 1;
            return res;
        }
        private static TensorLayout DeduceLayout(TensorLayout src){
            var res = new TensorLayout(src.Shape, DType.Float64);
            res.Shape.AsSpan().Fill(1);
            return res;
        }
    }

    public static partial class Tensor{
        /// <Meanmary>
        /// Mean the tensor.
        /// </Meanmary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The tensor to be mean. </param>
        /// <param name="axes"> The axes to mean. </param>
        /// <param name="keepDims"> Whether to keep the dims after the mean. If false, the NDim of the result may be different with the input. </param>
        /// <returns>The Mean tensor</returns>
        public static Tensor<double> Mean<T>(Tensor<T> src, int[] axes, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible{
            return src.Mean(axes, keepDims);
        }
        /// <Meanmary>
        /// Mean the tensor.
        /// </Meanmary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The tensor to be meaned. </param>
        /// <param name="axis"> The axis to mean. </param>
        /// <param name="keepDims"> Whether to keep the dims after the mean. If false, the NDim of the result may be different with the input. </param>
        /// <returns>The Mean tensor</returns>
        public static Tensor<double> Mean<T>(Tensor<T> src, int axis, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible{
            return src.Mean(axis, keepDims);
        }
        /// <Meanmary>
        /// Mean the tensor.
        /// </Meanmary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The tensor to be meaned. </param>
        /// <param name="keepDims"> Whether to keep the dims after the mean. If false, the NDim of the result may be different with the input. </param>
        /// <returns>The Mean tensor</returns>
        public static Tensor<double> Mean<T>(Tensor<T> src, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible{
            return src.Mean(keepDims);
        }
    }
}