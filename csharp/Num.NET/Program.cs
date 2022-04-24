using Numnet;
using Numnet.Algebra;
using Numnet.Manipulation;

Tensor<int> a = Tensor.FromArray<int>(new int[]{1, 2, 3, 4, 5, 6}, new int[] { 1, 3, 2 });
Tensor<double> b = Tensor.FromArray<double>(new double[]{1.2, 2.6, 3.9, 4.1, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6, 1.2, 2.6, 
                    3.9, 4.1, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6}, new int[] { 4, 2, 4 });
a.BroadcastTo(new int[]{4, 3, 2});
Console.WriteLine(a);

Tensor c = a.Matmul(b);
Console.WriteLine(c);
c = c.Reshape(new int[] { 6, 8 });
Console.WriteLine(c);
c = c.Permute(new int[] { 1, 0 });
Console.WriteLine(c);
c = c.Transpose(0, 1);
Console.WriteLine(c);
c[0, 0] = 999.0;
c[5, 7] = -888.0;
Console.WriteLine(c);
var s = c[1..3, ^5..^2];
Console.WriteLine(s);

Console.WriteLine(s.At<int>(0, 1));