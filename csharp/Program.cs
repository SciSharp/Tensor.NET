using System;
using System.Runtime.InteropServices;
using System.Buffers;
using Numnet.Native;
using Numnet;

unsafe{
    Tensor<int> a = new Tensor<int>(new ulong[] { 2, 3 });
    Tensor<int> b = new Tensor<int>(new ulong[] { 3, 4 });
    Tensor<int> c = new Tensor<int>(new ulong[] { 2, 4 });
    Console.WriteLine(a.TLayout._ndim);
    Console.WriteLine(b.TLayout._ndim);
    Console.WriteLine(c.TLayout._ndim);
    a.AsSpan().Fill(1);
    b.AsSpan().Fill(2);
    IntPtr status = Tensor<int>.Execute(a, b, c, NativeApi.Matmul, IntPtr.Zero, Provider.Naive);
    for (int i = 0; i < c.AsSpan().Length; i++){
        Console.Write(c.AsSpan()[i] + " ");
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