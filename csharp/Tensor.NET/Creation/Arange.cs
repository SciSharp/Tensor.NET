using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;

namespace Tensornet{
    public static partial class Tensor{
        public static Tensor<T> Arange<T>(double stop) where T : struct, IConvertible, IEquatable<T>{
            int num = (int)System.Math.Ceiling(stop);
            Tensor<T> res = new Tensor<T>(new TensorShape(num), TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            ArangeInternal<T>(res, 0, stop, 1);
            return res;
        }

        public static Tensor<T> Arange<T>(double start, double stop) where T : struct, IConvertible, IEquatable<T>{
            int num = (int)System.Math.Ceiling(stop - start);
            Tensor<T> res = new Tensor<T>(new TensorShape(num), TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            ArangeInternal<T>(res, start, stop, 1);
            return res;
        }

        public static Tensor<T> Arange<T>(double start, double stop, double step) where T : struct, IConvertible, IEquatable<T>{
            if(step <= 0){
                throw new InvalidParamException("The step of arrange should be a positive number.");
            }
            int num = (int)System.Math.Ceiling((stop - start) / step);
            Tensor<T> res = new Tensor<T>(new TensorShape(num), TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            ArangeInternal<T>(res, start, stop, step);
            return res;
        }

        private static unsafe void ArangeInternal<T>(Tensor<T> t, double start, double stop, double step) where T : struct, IConvertible, IEquatable<T>{
            ArangeParam param = new ArangeParam() { start = start, stop = stop, step = step};
            IntPtr status = NativeExecutor.Execute(NativeApi.Arange, t.TMemory, t.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
    }
}