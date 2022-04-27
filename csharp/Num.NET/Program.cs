using Numnet;
using Numnet.Algebra;
using Numnet.Manipulation;
using Numnet.Common;
using Numnet.Math;

Tensor<int> a = Tensor.FromArray<int>(new int[]{1, 2, 3, 4, 5, 6}, new int[] { 1, 3, 2 });
Tensor<double> b = Tensor.FromArray<double>(new double[]{1.2, 2.6, 3.9, 4.1, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6, 1.2, 2.6, 
                    3.9, 4.1, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6}, new int[] { 4, 2, 4 });
a.BroadcastTo(new int[]{4, 3, 2});
Console.WriteLine(a);

var c = a.Matmul(b);
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

Tensor<int> aa = Tensor.FromArray<int>(new int[]{1, 2, 3, 4, 5, 6}, new int[] { 3, 2 });
Tensor<int> bb = Tensor.FromArray<int>(new int[]{6, 5, 4, 3, 2, 1}, new int[] { 3, 2 });
var cs = s.ToTensor<int>();
Console.WriteLine(cs.Dim);
Console.WriteLine(cs);

var cc = InterElemOperation.Execute<int, int, int>(aa, bb, (x, y) => x * 2 - y);
Console.WriteLine(cc);
var nn = OnElemOperation.Execute<int, double>(cc, x => Math.Pow(x, 2));
Console.WriteLine(nn);

var mm = nn.Add(nn);
Console.WriteLine(mm);

Console.WriteLine(mm > new Temp(15));
