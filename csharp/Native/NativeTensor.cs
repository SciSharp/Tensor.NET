using System.Runtime.InteropServices;

namespace Numnet.Native{
    public enum DType: Int32
    {
        Invalid = 0,
        Int32 = 1,
        Float32 = 2,
        Float64 = 3,
        Bool = 4
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeTensor
    {
        public DType dtype;
        public int ndim;
        public ulong offset;
        public IntPtr shape;
        public IntPtr stride;
        public IntPtr data;
    }
}