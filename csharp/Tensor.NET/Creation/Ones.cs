using Tensornet.Common;
using Tensornet.Native.Param;
using Tensornet.Native;

namespace Tensornet{
    public static partial class Tensor{
        public static Tensor<T> Ones<T>(TensorShape shape) where T : struct, IConvertible, IEquatable<T>{
            Tensor<T> res = new Tensor<T>(new TensorLayout(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype));
            OnesInternal(res);
            return res;
        }
        private unsafe static void OnesInternal<T>(Tensor<T> src) where T : struct, IEquatable<T>, IConvertible{
            FillParam param = new FillParam() { value = 1.0 };
            IntPtr status = NativeExecutor.Execute(NativeApi.Fill, src.TMemory, src.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
        public static Tensor<TResult> OnesLike<TResult, TRefer>(Tensor<TRefer> tensor) 
            where TResult : struct, IConvertible, IEquatable<TResult> 
            where TRefer : struct, IConvertible, IEquatable<TRefer>{
            return Tensor.Ones<TResult>(tensor.TLayout);
        }
    }
}