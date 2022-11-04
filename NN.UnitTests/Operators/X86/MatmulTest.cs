using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NN.Native.Data;
using NN.Native.Basic;

namespace NN.UnitTests.Operators.X86
{
    public class MatmulTest
    {
        public (int[], int[])[] Shapes { get; set; } = new(int[], int[])[] {
                (new int[] {2, 4}, new int[] {4, 4}),
                (new int[] {3, 7}, new int[] {7, 16}),
                (new int[] {1, 20}, new int[] {20, 1}),
                (new int[] {10, 1}, new int[] {1, 20}),
                (new int[] {100, 100}, new int[] {100, 101}),
            };

        [Fact]
        public unsafe void Float32Test()
        {
            foreach(var shape in Shapes)
            {
                var a = NativeArray.Random.Normal<int>(new NativeLayout(shape.Item1), -10, 10);
                var b = NativeArray.Random.Normal<int>(new NativeLayout(shape.Item2), -10, 10);
                var naiveResult = new NativeArray<int>(new NativeLayout(new int[] { shape.Item1[0], shape.Item2[1] }), new DefaultNativeMemoryManager());
                var x86Result = new NativeArray<int>(new NativeLayout(new int[] { shape.Item1[0], shape.Item2[1] }), new DefaultNativeMemoryManager());
                Native.Operators.Naive.MatmulOperator<int>.Exec(a.Span, b.Span, naiveResult.Span, a._layout, b._layout, naiveResult._layout);
                Native.Operators.X86.MatmulOperator<int>.Exec(a.Span, b.Span, x86Result.Span, a._layout, b._layout, x86Result._layout);
                var enumeratorNaive = NativeLayout.GetIndexEnumerator(naiveResult._layout);
                var enumeratorX86 = NativeLayout.GetIndexEnumerator(x86Result._layout);
                var naiveSpan = naiveResult.Span;
                var x86SPan = x86Result.Span;
                for(int i = 0; i < naiveResult._layout.TotalElemCount(); i++)
                {
                    Assert.Equal(naiveSpan[enumeratorNaive.MoveNext()], x86SPan[enumeratorX86.MoveNext()]);
                }
            }
        }
    }
}
