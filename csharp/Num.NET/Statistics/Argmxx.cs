using Numnet.Common;
using Numnet.Native;
using Numnet.Exceptions;
using Numnet.Native.Param;

namespace Numnet{
    public static class ArgmaxExtension{

        public static Tensor<long> Argmax<T>(this Tensor<T> src, int axis) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<long> res = new Tensor<long>(DeduceLayout(src.TLayout, axis));
            res.TLayout.InitContiguousLayout();
            ArgmxxInternal(src, res, axis, true);
            res.TLayout.RemoveAxisInplace(axis);
            return res;
        }

        public static Tensor<long> Argmin<T>(this Tensor<T> src, int axis) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<long> res = new Tensor<long>(DeduceLayout(src.TLayout, axis));
            res.TLayout.InitContiguousLayout();
            ArgmxxInternal(src, res, axis, false);
            res.TLayout.RemoveAxisInplace(axis);
            return res;
        }

        private unsafe static void ArgmxxInternal<T>(Tensor<T> src, Tensor<long> dst, int axis, bool isMax) where T : struct, IEquatable<T>, IConvertible{
            ArgmxxParam param = new ArgmxxParam() { axis = axis, isMax = isMax };
            IntPtr status = NativeExecutor.Execute(NativeApi.Argmxx, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
        private static TensorLayout DeduceLayout(TensorLayout src, int axis){
            TensorLayout res = new TensorLayout();
            if (axis < 0 || axis >= src.NDim) {
                throw new InvalidParamException("Invalid argmxx axis param.");
            }
            res.DType = DType.Int64;
            res.NDim = src.NDim;
            for (int i = 0; i < src.NDim; i++) {
                res.Shape[i] = src.Shape[i];
            }
            res.Shape[axis] = 1;
            return res;
        }
    }

    public static partial class Tensor{
        public static Tensor<long> Argmax<T>(Tensor<T> src, int axis) where T : struct, IEquatable<T>, IConvertible{
            return src.Argmax(axis);
        }
        public static Tensor<long> Argmin<T>(Tensor<T> src, int axis) where T : struct, IEquatable<T>, IConvertible{
            return src.Argmin(axis);
        }
    }
}