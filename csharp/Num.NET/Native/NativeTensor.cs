using System.Runtime.InteropServices;

namespace Numnet.Native{
    public enum NativeProvider: Int32 { Naive = 0, ST_x86 = 1, MT_x86 = 2 };
    public enum DType: Int32
    {
        Invalid = 0,
        Int32 = 1,
        Float32 = 2,
        Float64 = 3,
        Int64 = 4,
        Bool = 5
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeTensor
    {
        public DType dtype;
        public int ndim;
        public int offset;
        public IntPtr shape;
        public IntPtr stride;
        public IntPtr data;
    }
}