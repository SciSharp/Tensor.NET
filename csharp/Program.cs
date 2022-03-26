using System;
using System.Runtime.InteropServices;
using System.Buffers;
using Numnet.Native;
using Numnet;

unsafe{
    Tensor<int> a = new Tensor<int>(new ulong[] { 2, 3 });
    Tensor<int> b = new Tensor<int>(new ulong[] { 3, 4 });
    Tensor<int> c = new Tensor<int>(new ulong[] { 2, 4 });
    Console.WriteLine(a._layout._ndim);
    Console.WriteLine(b._layout._ndim);
    Console.WriteLine(c._layout._ndim);
    a._dataHandle.AsSpan<int>().Fill(1);
    b._dataHandle.AsSpan<int>().Fill(2);
    IntPtr status = Tensor<int>.Execute(a, b, c, NativeApi.Matmul, IntPtr.Zero);
    for (int i = 0; i < c._dataHandle.AsSpan<int>().Length; i++){
        Console.Write(c._dataHandle.AsSpan<int>()[i] + " ");
    }
    Console.WriteLine();
    Console.WriteLine(NativeStatus.GetErrorCode(status));
    Console.WriteLine(NativeStatus.GetErrorMessage(status));
}


[StructLayout(LayoutKind.Sequential)]
struct Shape{
    public IntPtr shape;
    public ulong ndim;
}