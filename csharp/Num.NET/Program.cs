using System.Runtime.InteropServices;
using Numnet.Native;
using Numnet.Tensor;
using Numnet.Tensor.Utilities;
using Numnet.Native.Param;

unsafe{
    Tensor<int> a = new Tensor<int>(new int[]{1, 2, 3, 4, 5, 6}, new int[] { 1, 3, 2 });
    Tensor<double> b = new Tensor<double>(new double[]{1.2, 2.6, 3.9, 4.1, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6, 1.2, 2.6, 
                        3.9, 4.1, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6}, new int[] { 4, 2, 4 });
    a = a.BroadCast(new int[]{4, 3, 2});
    Console.WriteLine(a);
    int[] permuteParam = new int[3] { 2, 0, 1 };
    var ap = Tensor.Zeros<int>(new int[] { 2, 4, 3 });
    fixed(int* p = permuteParam){
        var param = new Permute() { dims = new IntPtr(p) };
        var ptr = new IntPtr(&param);
        IntPtr st = Tensor<int>.Execute(a, ap, NativeApi.Permute, ptr, Provider.Naive);
        Console.WriteLine(NativeStatus.GetErrorCode(st));
        Console.WriteLine(NativeStatus.GetErrorMessage(st));
    }
    Console.WriteLine(ap);
    Tensor<double> c = Tensor.Zeros<double>(new int[] { 4, 3, 4 });
    IntPtr status = Tensor<int>.Execute(a, b, c, NativeApi.Matmul, IntPtr.Zero, Provider.Naive);
    Console.WriteLine(NativeStatus.GetErrorCode(status));
    Console.WriteLine(NativeStatus.GetErrorMessage(status));
    Console.WriteLine(c);
    Console.WriteLine(c.Reshape(new int[] { 6, 8 }));
}