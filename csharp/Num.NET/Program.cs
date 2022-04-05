using System.Runtime.InteropServices;
using Numnet.Native;
using Numnet.Tensor;
using Numnet.Tensor.Utilities;

unsafe{
    Tensor<int> a = new Tensor<int>(new int[]{1, 2, 3, 4, 5, 6}, new int[] { 1, 3, 2 });
    Tensor<double> b = new Tensor<double>(new double[]{1.2, 2.6, 3.9, 4.1, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6, 1.2, 2.6, 
                        3.9, 4.1, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6}, new int[] { 3, 2, 4 });
    a = a.BroadCast(new int[]{3, 3, 2});
    Console.WriteLine(a);
    Tensor<double> c = Tensor.Zeros<double>(new int[] { 3, 3, 4 });
    IntPtr status = Tensor<int>.Execute(a, b, c, NativeApi.Matmul, IntPtr.Zero, Provider.Naive);
    Console.WriteLine(NativeStatus.GetErrorCode(status));
    Console.WriteLine(NativeStatus.GetErrorMessage(status));
    Console.WriteLine(c);
    Console.WriteLine(c.Reshape(new int[] { 4, 9 }));
}