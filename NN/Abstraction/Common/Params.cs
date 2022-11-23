using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.Common
{
    public readonly record struct MatmulParam(
        bool TransposeA,
        bool TransposeB
    );
    public readonly record struct TypeConvertParam(
        bool Transpose,
        int DimA,
        int DimB
    );
    public readonly record struct TransposeParam(
        int DimA,
        int DimB
    );
    public readonly ref struct ReduceParam
    {
        public readonly ReadOnlySpan<int> Dims;
        public ReduceParam(ReadOnlySpan<int> dims)
        {
            Dims = dims;
        }
        public ReduceParam(params int[] dims)
        {
            Dims = dims.AsSpan();
        }
    }
}
