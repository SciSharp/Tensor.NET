using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Native.Param;

namespace Tensornet{
    public static partial class Tensor{
        public static Tensor<T> Eye<T>(int rows, int cols, int k) where T : struct, IConvertible, IEquatable<T>{
            Tensor<T> res = new Tensor<T>(new TensorShape(rows, cols), TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            EyeInternal<T>(res, k);
            return res;
        }

        private static unsafe void EyeInternal<T>(Tensor<T> t, int k) where T : struct, IConvertible, IEquatable<T>{
            EyeParam param = new EyeParam() { k = k };
            IntPtr status = NativeExecutor.Execute(NativeApi.Eye, t.TMemory, t.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
    }
}