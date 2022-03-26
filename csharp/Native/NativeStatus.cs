using System.Runtime.InteropServices;

namespace Numnet.Native{
    internal enum StatusCode: Int32
    {
        OK = 0,
        FAIL = 1,
        INVALID_ARGUMENT = 2,
        MISMATCHED_SHAPE = 3,
        MISMATCHED_DTYPE = 4,
        ENGINE_ERROR = 5,
        RUNTIME_EXCEPTION = 6,
        INVALID_PROTOBUF = 7,
        INVALID_PARAM = 8,
        NOT_IMPLEMENTED = 9
    };

    internal static class NativeStatus{
        public static void AssertOK(){

        }

        public static StatusCode GetErrorCode(IntPtr status){
            Console.WriteLine("Get!");
            return status == IntPtr.Zero?StatusCode.OK : NativeApi.GetErrorCode(status);
        }

        public static string GetErrorMessage(IntPtr status){
            if(status == IntPtr.Zero){
                return "OK";
            }
            string? res = Marshal.PtrToStringUTF8(NativeApi.GetErrorMessage(status));
            if(res is null){
                return "No error message.";
            }
            return res;
        }
    }
}