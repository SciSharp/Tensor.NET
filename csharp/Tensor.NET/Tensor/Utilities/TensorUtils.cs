using Tensornet.Native;

namespace Tensornet{
    public static class TensorUtils
    {
        /// <summary>
        /// Return if each value of the tensors are equal.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="err"></param>
        /// <returns></returns>
        [Obsolete("The method may be removed in the future version")]
        public static bool IsValueEqual<T>(Tensor<T> a, Tensor<T> b, double err = 1e-5) where T : struct, IEquatable<T>, IConvertible{
            if(Object.ReferenceEquals(a, b)) return true;
            if(!a.TLayout.IsSameShape(b.TLayout)) return false;
            int[] indices = new int[a.TLayout.NDim];
            for (int i = 0; i < a.TLayout.TotalElemCount(); i++){
                if(System.Math.Abs(Convert.ToDouble(a[indices]) - Convert.ToDouble(b[indices])) > err){
                    return false;
                }
                indices[a.TLayout.NDim - 1]++;
                for (int j = a.TLayout.NDim - 1; j >= 1; j--){
                    if(indices[j] == a.TLayout.Shape[j]){
                        indices[j - 1]++;
                        indices[j] = 0;
                    }
                }
            }
            return true;
        }
    }
}