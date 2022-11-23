using NN.Native.Basic;
using NN.Native.Data;
using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Extensions
{
    public static class ArrayEqualExtension
    {
        public static bool IsElementsEqualWith(this NativeArray<float> array, NativeArray<float> other)
        {
            return AreEqual(array, other, (x, y) => MathF.Abs(x - y) < 0.01f);
        }
        public static bool IsElementsEqualWith(this NativeArray<double> array, NativeArray<double> other)
        {
            return AreEqual(array, other, (x, y) => Math.Abs(x - y) < 0.0001);
        }
        public static bool IsElementsEqualWith(this NativeArray<int> array, NativeArray<int> other)
        {
            return AreEqual(array, other, (x, y) => x == y);
        }
        private static bool AreEqual<T>(in NativeArray<T> a, in NativeArray<T> b, Func<T, T, bool> equalFunc) where T: unmanaged
        {
            if(!a._layout.IsSameShape(b._layout))
            {
                return false;
            }
            var totalCount = a._layout.TotalElemCount();
            
            var enumeratorA = NativeLayout.GetIndexEnumerator(a._layout);
            var enumeratorB = NativeLayout.GetIndexEnumerator(b._layout);
            var spanA = a.Span;
            var spanB = b.Span;
            for(int i = 0; i < totalCount; i++)
            {
                if (!equalFunc(spanA[enumeratorA.MoveNext()], spanB[enumeratorB.MoveNext()]))
                {
                    Debug.WriteLine($"Two arrays differ at offset {i}. {a._layout.GetOffsetDescription(i)}");
                    return false;
                }
            }
            return true;
        }
    }
}
