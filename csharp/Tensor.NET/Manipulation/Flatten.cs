namespace Tensornet{
    public static class FlattenExtension{
        /// <summary>
        /// Flatten the tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <returns></returns>
        public static Tensor<T> Flatten<T>(this Tensor<T> src) where T : struct, IEquatable<T>, IConvertible
        {
            return src.Reshape(new int[] { src.TLayout.TotalElemCount() });
        }
    }

    public static partial class Tensor{
        /// <summary>
        /// Flatten the tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <returns></returns>
        public static Tensor<T> Flatten<T>(Tensor<T> src) where T : struct, IEquatable<T>, IConvertible
        {
            return src.Reshape(new int[] { src.TLayout.TotalElemCount() });
        }
    }
}