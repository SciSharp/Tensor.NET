using Numnet.Common;
using Numnet.Native;
using Numnet.Native.Param;

namespace Numnet{
    public static partial class Tensor{
        public static Tensor<T> Linspace<T>(double start, double stop, int num, bool isEndPoint = true) where T : struct, IConvertible, IEquatable<T>{
            Tensor<T> res = new Tensor<T>(new TensorShape(num), TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            LinspaceInternal<T>(res, start, stop, num, isEndPoint);
            return res;
        }

        private static unsafe void LinspaceInternal<T>(Tensor<T> t, double start, double stop, int num, bool isEndPoint) where T : struct, IConvertible, IEquatable<T>{
            LinspaceParam param = new LinspaceParam() { start = start, stop = stop, num = num, isEndpoint = isEndPoint };
            IntPtr status = NativeExecutor.Execute(NativeApi.Linspace, t.TMemory, t.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
    }
}