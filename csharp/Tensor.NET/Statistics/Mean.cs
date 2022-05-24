using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;

namespace Tensornet{
    public static class MeanExtension{
        /// <summary>
        /// Get the mean of elements of a tensor along some axes.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="axes"> The axes to get mean. </param>
        /// <param name="keepDims"> Whether to keep the dims all eliminate the dims. False by default. </param>
        /// <returns></returns>
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
        /// <summary>
        /// Get the mean of elements of a tensor along an axis.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="axis"> The axis to get mean. </param>
        /// <param name="keepDims"> Whether to keep the dims all eliminate the dims. False by default. </param>
        /// <returns></returns>
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
        /// <summary>
        /// Get the mean of elements of a tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="keepDims"> Whether to keep the dims all eliminate the dims. False by default. </param>
        /// <returns></returns>
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
                MeanParam p = new MeanParam() { dims = new IntPtr(ptr) };
                IntPtr status = NativeExecutor.Execute(NativeApi.Mean, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&p), Tensor<T>.Provider);
                NativeStatus.AssertOK(status);
            }
            if(!keepDims){
                dst.TLayout.RemoveAllDanglingAxisInplace();
            }
        }
        private static TensorLayout DeduceLayout(TensorLayout src, int[] axes){
            var res = new TensorLayout(src as TensorShape, DType.Float64);
            foreach(var dim in axes){
                res.Shape[dim] = 1;
            }
            return res;
        }
        private static TensorLayout DeduceLayout(TensorLayout src, int axis){
            var res = new TensorLayout(src as TensorShape, DType.Float64);
            res.Shape[axis] = 1;
            return res;
        }
        private static TensorLayout DeduceLayout(TensorLayout src){
            var res = new TensorLayout(src as TensorShape, DType.Float64);
            res.Shape.AsSpan().Fill(1);
            return res;
        }
    }

    public static partial class Tensor{
        /// <Meanmary>
        /// Get the mean of the tensor along some axes.
        /// </Meanmary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> </param>
        /// <param name="axes"> The axes to get mean result. </param>
        /// <param name="keepDims"> Whether to keep the dims all eliminate the dims. False by default. </param>
        /// <returns>The Mean tensor</returns>
        public static Tensor<double> Mean<T>(Tensor<T> src, int[] axes, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible{
            return src.Mean(axes, keepDims);
        }
        /// <Meanmary>
        /// Get the mean of the tensor along an axis.
        /// </Meanmary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="axis"> The axis to get mean result. </param>
        /// <param name="keepDims"> Whether to keep the dims all eliminate the dims. False by default. </param>
        /// <returns>The Mean tensor</returns>
        public static Tensor<double> Mean<T>(Tensor<T> src, int axis, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible{
            return src.Mean(axis, keepDims);
        }
        /// <Meanmary>
        /// Mean the tensor.
        /// </Meanmary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> </param>
        /// <param name="keepDims"> Whether to keep the dims all eliminate the dims. False by default. </param>
        /// <returns>The Mean tensor</returns>
        public static Tensor<double> Mean<T>(Tensor<T> src, bool keepDims = false) where T : struct, IEquatable<T>, IConvertible{
            return src.Mean(keepDims);
        }
    }
}