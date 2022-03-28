using System.Runtime.InteropServices;

namespace Numnet.Native{
    internal static class NativeApi{
        [DllImport("/home/lyh/code/Num.NET/build/apis/libnumnet.so")]
        public static extern StatusCode GetErrorCode(IntPtr status);
        [DllImport("/home/lyh/code/Num.NET/build/apis/libnumnet.so")]
        public static extern IntPtr GetErrorMessage(IntPtr status);
        [DllImport("/home/lyh/code/Num.NET/build/apis/libnumnet.so")]
        public static extern void FreeStatusMemory(IntPtr status);


        [DllImport("/home/lyh/code/Num.NET/build/apis/libnumnet.so")]
        unsafe public static extern IntPtr Matmul(NativeTensor* a, NativeTensor* b, NativeTensor* oup, IntPtr param, Provider provider);
        
    }
}