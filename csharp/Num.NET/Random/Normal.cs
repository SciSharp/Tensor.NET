using Numnet.Common;
using Numnet.Native;
using Numnet.Native.Param;

namespace Numnet{
    public static partial class Tensor{
        public static Tensor<T> Normal<T>(TensorShape shape, double mean = .0, double std = .1) where T : struct, IConvertible, IEquatable<T>{
            Tensor<T> res = new Tensor<T>(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            NormalInternal<T>(res, mean, std);
            return res;
        }
        private unsafe static void NormalInternal<T>(Tensor<T> t, double mean, double std) where T : struct, IConvertible, IEquatable<T>{
            NormalParam param = new NormalParam() { mean = mean, std = std };
            IntPtr status = NativeExecutor.Execute(NativeApi.Normal, t.TMemory, t.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
    }
}