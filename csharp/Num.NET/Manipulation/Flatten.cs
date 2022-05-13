namespace Numnet{
    public static class FlattenExtension{

        public static Tensor<T> Flatten<T>(this Tensor<T> src) where T : struct, IEquatable<T>, IConvertible
        {
            return src.Reshape(new int[] { src.TLayout.TotalElemCount() });
        }
    }

    public static partial class Tensor{
        public static Tensor<T> Flatten<T>(Tensor<T> src) where T : struct, IEquatable<T>, IConvertible
        {
            return src.Reshape(new int[] { src.TLayout.TotalElemCount() });
        }
    }
}