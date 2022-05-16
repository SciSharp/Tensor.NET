using System.Runtime.InteropServices;

namespace Tensornet.Native{
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
        /// <summary>
        /// Assert that the status indicates no error. Note that the memory that "status" points to will be free after this call.
        /// </summary>
        /// <param name="status"></param>
        /// <exception cref="Tensornet.Exceptions.NNRuntimeException"></exception>
        public static void AssertOK(IntPtr status){
            if(status == IntPtr.Zero){
                return;
            }
            var errorCode = GetErrorCode(status);
            var errorMsg = GetErrorMessage(status);
            DisposeStatus(status);
            throw new Tensornet.Exceptions.NNRuntimeException($"A runtime error occured, [{Enum.GetName(typeof(StatusCode), errorCode)}]: {errorMsg}");
        }

        public static void DisposeStatus(IntPtr status){
            NativeApi.FreeStatusMemory(status);
        }


        public static StatusCode GetErrorCode(IntPtr status){
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