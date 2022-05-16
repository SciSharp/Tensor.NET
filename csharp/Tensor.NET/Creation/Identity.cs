using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Native.Param;

namespace Tensornet{
    public static partial class Tensor{
        public static Tensor<T> Idendity<T>(int n) where T : struct, IConvertible, IEquatable<T>{
            return Tensor.Eye<T>(n, n, 0);
        }
    }
}