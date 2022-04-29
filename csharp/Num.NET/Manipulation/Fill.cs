using Numnet.Common;
using Numnet.Native;
using Numnet.Exceptions;
using Numnet.Native.Param;

namespace Numnet.Manipulation{
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
    }
}