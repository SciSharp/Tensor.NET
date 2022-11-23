using NN.Native.Operators.Naive;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Core
{
    public partial class NDArray<T>
    {
        public static NDArray<double> Pow(NDArray<double> a, double b)
        {
#if NET7_0_OR_GREATER
            var res = TernaryElemWiseOperator<double, double, double>.Exec(a._nativeArray, b, (x, y, idx) => Math.Pow(x, y));
#else
            var res = new TernaryElemWiseOperator<double, double, double>().Exec(a._nativeArray, b, (x, y, idx) => Math.Pow(x, y));
#endif
            return NDArray<double>.FromNativeArray(res);
        }
        public static NDArray<double> Pow(double a, NDArray<double> b)
        {
#if NET7_0_OR_GREATER
            var res = TernaryElemWiseOperator<double, double, double>.Exec(a, b._nativeArray, (x, y, idx) => Math.Pow(x, y));
#else
            var res = new TernaryElemWiseOperator<double, double, double>().Exec(a, b._nativeArray, (x, y, idx) => Math.Pow(x, y));
#endif
            return NDArray<double>.FromNativeArray(res);
        }

        public static NDArray<double> Log2(NDArray<double> a)
        {
#if NET7_0_OR_GREATER
            var res = BinaryElemWiseOperator<double, double>.Exec(a._nativeArray, (x, idx) => Math.Log2(x));
#else
            var res = new BinaryElemWiseOperator<double, double>().Exec(a._nativeArray, (x, idx) => Math.Log2(x));
#endif
            return NDArray<double>.FromNativeArray(res);
        }
    }
}
