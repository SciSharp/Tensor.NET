using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Native.Param;

namespace Tensornet{
    public static partial class Tensor{
        /// <summary>
        /// Return the identity tensor, which is a square two-dimensional tensor with ones on the main diagonal.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.identity.html#numpy.identity
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="n"> Number of rows (and columns). </param>
        /// <returns></returns>
        public static Tensor<T> Idendity<T>(int n) where T : struct, IConvertible, IEquatable<T>{
            return Tensor.Eye<T>(n, n, 0);
        }
    }
}