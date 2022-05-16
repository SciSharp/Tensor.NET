using System.Runtime.InteropServices;

namespace Tensornet.Native{
    internal static class NativeApi{
        [DllImport("libnumnet")]
        public static extern StatusCode GetErrorCode(IntPtr status);
        [DllImport("libnumnet")]
        public static extern IntPtr GetErrorMessage(IntPtr status);
        [DllImport("libnumnet")]
        public static extern void FreeStatusMemory(IntPtr status);


        [DllImport("libnumnet")]
        public static extern IntPtr Matmul(IntPtr a, IntPtr b, IntPtr oup, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Dot(IntPtr a, IntPtr b, IntPtr oup, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr BoolIndex(IntPtr a, IntPtr b, IntPtr oup, IntPtr param, NativeProvider provider);


        [DllImport("libnumnet")]
        public static extern IntPtr Permute(IntPtr inp, IntPtr oup, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Transpose(IntPtr inp, IntPtr oup, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Argmxx(IntPtr inp, IntPtr oup, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Repeat(IntPtr inp, IntPtr oup, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Flip(IntPtr inp, IntPtr oup, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr MatrixInverse(IntPtr inp, IntPtr oup, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Rotate(IntPtr inp, IntPtr oup, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Pad(IntPtr inp, IntPtr oup, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Sort(IntPtr inp, IntPtr oup, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Onehot(IntPtr inp, IntPtr oup, IntPtr param, NativeProvider provider);


        [DllImport("libnumnet")]
        public static extern IntPtr TypeConvert(IntPtr inp, IntPtr oup, IntPtr param, NativeProvider provider);

        [DllImport("libnumnet")]
        public static extern IntPtr Concat(IntPtr inps, int size, IntPtr oup, IntPtr param, NativeProvider provider);

        [DllImport("libnumnet")]
        public static extern IntPtr Normal(IntPtr t, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Uniform(IntPtr t, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Eye(IntPtr t, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Fill(IntPtr t, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Arange(IntPtr t, IntPtr param, NativeProvider provider);
        [DllImport("libnumnet")]
        public static extern IntPtr Linspace(IntPtr t, IntPtr param, NativeProvider provider);
    }
}