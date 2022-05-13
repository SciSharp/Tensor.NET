using Numnet.Common;
using Numnet.Native;
using Numnet.Native.Param;

namespace Numnet{
    public static partial class Tensor{
        [Obsolete("Not implemented", true)]
        public static Tensor<T> Onehot<T>(int rows, int cols, int k) where T : struct, IConvertible, IEquatable<T>{
            Tensor<T> res = new Tensor<T>(new TensorShape(rows, cols), TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            EyeInternal<T>(res, k);
            return res;
        }
    }
}