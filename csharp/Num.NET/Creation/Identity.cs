using Numnet.Common;
using Numnet.Native;
using Numnet.Native.Param;

namespace Numnet{
    public static partial class Tensor{
        public static Tensor<T> Idendity<T>(int n) where T : struct, IConvertible, IEquatable<T>{
            return Tensor.Eye<T>(n, n, 0);
        }
    }
}