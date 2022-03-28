using System;
using System.Runtime.InteropServices;
using System.Buffers;
using Numnet.Native;
using Numnet.Tensor;
using Numnet.Tensor.Utilities;

unsafe{
    Tensor<int> a = new Tensor<int>(new int[]{1, 2, 3, 4, 5, 6}, new int[] { 2, 3 });
    Tensor<double> b = new Tensor<double>(new double[]{1.2, 2.6, 3.9, 4.1, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6}, new int[] { 3, 4 });
    Tensor<double> c = Tensor.Zeros<double>(new int[] { 2, 4 });
    // b.AsSpan().Fill(1);
    // b.AsSpan()[1] = 2;
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