using Numnet.Native;

namespace Numnet{
    public static class TensorUtils
    {
        public static bool IsValueEqual(Tensor a, Tensor b, double err = 1e-5){
            if(Object.ReferenceEquals(a, b)) return true;
            if(!a.TLayout.IsSameShape(b.TLayout)) return false;
            if(a.TLayout.DType != b.TLayout.DType) return false;
            int[] indices = new int[a.TLayout.NDim];
            for (int i = 0; i < a.TLayout.TotalElemCount(); i++){
                bool equal = a.TLayout.DType switch
                {
                    DType.Int32 or DType.Int64 or DType.Bool => a[indices].Equals(b[indices]),
                    DType.Float32 => Math.Abs((float)a[indices] - (float)b[indices]) < err,
                    DType.Float64 => Math.Abs((double)a[indices] - (double)b[indices]) < err,
                    _ => false
                };
                if(!equal) return false;
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