using Numnet.Common;
using Numnet.Native.Param;
using Numnet.Native;

namespace Numnet{
    public static partial class Tensor{
        public static Tensor<T> Zeros<T>(TensorShape shape) where T : struct, IConvertible, IEquatable<T>{
            Tensor<T> res = new Tensor<T>(new TensorLayout(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype));
            ZerosInternal(res);
            return res;
        }
        private unsafe static void ZerosInternal<T>(Tensor<T> src) where T : struct, IEquatable<T>, IConvertible{
            FillParam param = new FillParam() { value = .0 };
            IntPtr status = NativeExecutor.Execute(NativeApi.Fill, src.TMemory, src.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
        public static Tensor<TResult> ZerosLike<TResult, TRefer>(Tensor<TRefer> tensor) 
            where TResult : struct, IConvertible, IEquatable<TResult> 
            where TRefer : struct, IConvertible, IEquatable<TRefer>{
            return Tensor.Zeros<TResult>(tensor.TLayout);
        }
    }
}