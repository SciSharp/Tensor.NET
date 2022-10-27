using NN.Native.Abstraction.DType;
using NN.Native.Basic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Operators.Naive
{
    public class MatmulOperator<TA, TB, TC, THandler> where TA: unmanaged where TB: unmanaged where TC: unmanaged where THandler: ITernaryDTypeHandler<TA, TB, TC>, IUnaryDTypeHandler<TC>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Exec(ReadOnlySpan<TA> a, ReadOnlySpan<TB> b, Span<TC> c, in NativeLayout layoutA, in NativeLayout layoutB, in NativeLayout layoutC)
        {
            // The array should be contiguous here
            int aRows = layoutA._shape[0];
            int aCols = layoutA._shape[1];
            int bCols = layoutB._shape[1];
            for(int i = 0; i < aRows; i++)
            {
                for(int j = 0; j < bCols; j++)
                {
                    var res = THandler.Zero;
                    for(int k = 0; k < aCols; k++)
                    {
                        res = THandler.MultiplyAndAdd(a[i * layoutA._shape[1] + k], b[k * layoutB._shape[1] + j], res);
                    }
                    c[i * layoutC._shape[1] + j] = res;
                }
            }
        }
    }
}
