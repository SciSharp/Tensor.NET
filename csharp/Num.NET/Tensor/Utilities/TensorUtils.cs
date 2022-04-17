namespace Numnet{
    public static class TensorUtils
    {
        public static bool IsValueEqual(Tensor a, Tensor b){
            if(Object.ReferenceEquals(a, b)) return true;
            if(!a.TLayout.IsSameShape(b.TLayout)) return false;
            if(a.TLayout.DType != b.TLayout.DType) return false;
            int[] indices = new int[a.TLayout.NDim];
            for (int i = 0; i < a.TLayout.TotalElemCount(); i++){
                Console.WriteLine(string.Join(", ", indices));
                if (!a[indices].Equals(b[indices]))
                {
                    Console.WriteLine($"exit, a: {a[indices]}, b: {b[indices]}");
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