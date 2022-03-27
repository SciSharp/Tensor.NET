using System.Buffers;
using System.Runtime.InteropServices;

namespace Numnet.Tensor.Utilities{
    public static partial class Tensor
    {
        public static Tensor<T> Zeros<T>(int[] shape) where T:struct{
            Tensor<T> res = new Tensor<T>(shape);
            MemoryMarshal.Cast<T, byte>(res.TMemory.AsSpan()).Fill(0);
            return res;
        }
    }
}