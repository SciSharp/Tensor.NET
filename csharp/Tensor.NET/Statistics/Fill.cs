using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;

namespace Tensornet{
    public static class FillExtension{

        public static void Fill<T>(this Tensor<T> src, T value) where T : struct, IEquatable<T>, IConvertible
        {
            FillParam param = new FillParam() { value = Convert.ToDouble(value) };
            FillInternal<T>(src, value);
        }
        private unsafe static void FillInternal<T>(Tensor<T> src, T value) where T : struct, IEquatable<T>, IConvertible{
            FillParam param = new FillParam() { value = Convert.ToDouble(value) };
            IntPtr status = NativeExecutor.Execute(NativeApi.Fill, src.TMemory, src.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
        public static Tensor<TResult> FillLike<TResult, TRefer>(Tensor<TRefer> tensor, TResult value) 
            where TResult : struct, IConvertible, IEquatable<TResult> 
            where TRefer : struct, IConvertible, IEquatable<TRefer>{
            Tensor<TResult> res = new Tensor<TResult>(tensor.TLayout);
            res.Fill(value);
            return res;
        }
    }

    public static partial class Tensor{
        public static void Fill<T>(Tensor<T> src, T value) where T : struct, IEquatable<T>, IConvertible
        {
            src.Fill(value);
        }
    }
}