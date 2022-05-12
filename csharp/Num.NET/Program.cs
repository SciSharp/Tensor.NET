using Numnet;
using Numnet.Common;
using Numnet.Math;
using System.Text;

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

var gaussian = Tensor.Random.Normal<int>(new TensorShape(5, 6), 0, 5);
Console.WriteLine(gaussian);
var avg = Tensor.Random.Uniform<int>(new TensorShape(5, 6), -5, 5);
Console.WriteLine(avg);

var eye = Tensor.Eye<int>(4, 5, 2);
Console.WriteLine(eye);

eye.Fill(258);
Console.WriteLine(eye);

var linspace = Tensor.Linspace<double>(0, 10, 5);
Console.WriteLine(linspace);

var ones = Tensor.Ones<int>(new int[] { 2, 3 });
Console.WriteLine(ones);
ones = ones.Unsqueeze(1);
Console.WriteLine(ones);
ones = ones.Unsqueeze(3);
Console.WriteLine(ones);
ones = ones.Squeeze();
Console.WriteLine(ones);

var e = Tensor.Eye<int>(4, 5, 1);
Console.WriteLine(e);
var eList = e.ToList();
StringBuilder builder = new StringBuilder();
foreach(var item in eList){
    builder.Append($"{item} ");
}
Console.WriteLine(builder.ToString());

var ta = Tensor.Random.Uniform<int>(new int[] { 2, 3, 3 }, 0, 50);
var tb = Tensor.Random.Uniform<int>(new int[] { 2, 2, 3 }, 0, 50);
var tc = Tensor.Random.Uniform<int>(new int[] { 2, 4, 3 }, 0, 50);
var tconcat = Tensor.Concat(new Tensor<int>[] { ta, tb, tc }, 1);
Console.WriteLine(ta);
Console.WriteLine(tb);
Console.WriteLine(tc);
Console.WriteLine(tconcat);

var bidx = Tensor.Random.Uniform<bool>(new int[] { 3, 3 }, 0, 1);
var bres = ta[ta > 20];
Console.WriteLine(ta);
Console.WriteLine(bidx);
Console.WriteLine(bres);

var twhere = ta.Where(x => x > 20, 999, -555);
Console.WriteLine(twhere);
var trefer = Tensor.Random.Uniform<int>(new int[] { 2, 3, 3 }, -100, -50);
ta[ta > 20] = trefer;
Console.WriteLine(ta);

var da = Tensor.Random.Uniform<int>(new int[] { 2, 3 }, 0, 10);
var db = Tensor.Random.Uniform<int>(new int[] { 2, 3 }, 0, 10);
var dres = da.Dot(db);
Console.WriteLine(da);
Console.WriteLine(db);
Console.WriteLine(dres);

// var ts = Tensor.Random.Uniform<int>(new int[] { 2, 3 }, 0, 20);
// TensorWriter.Write("test.nn", ts);
// var tr = TensorReader.Read<float>("test.nn");
// Console.WriteLine(ts);
// Console.WriteLine(tr);

var src = Tensor.Random.Uniform<int>(new int[] { 4, 5 }, 0, 50);
var amax = src.Argmax(0);
var amin = src.Argmin(1);
Console.WriteLine(src);
Console.WriteLine(amax);
Console.WriteLine(amin);

var repeat = src.Repeat(2, 1);
Console.WriteLine(repeat);

var flip = src.Flip(0, 1);
Console.WriteLine(flip);

src = Tensor.Random.Uniform<int>(new int[] { 3, 3 }, 0, 20);
var inverse = src.Inverse();
Console.WriteLine(src);
Console.WriteLine(inverse);

src = Tensor.Random.Uniform<int>(new int[] { 3, 4 }, 10, 50);
var rotate = src.Rotate(3, 0, 1);
Console.WriteLine(src);
Console.WriteLine(rotate);

var constantPadding = src.Pad(new (int, int)[] { (1, 2), (1, 1) }, PadMode.Constant, new (double, double)[] { (1, 2), (3, 4) });
Console.WriteLine(constantPadding);
var edgePadding = src.Pad(1, PadMode.Edge, 0);
Console.WriteLine(edgePadding);