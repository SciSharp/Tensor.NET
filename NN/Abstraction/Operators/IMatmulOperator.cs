using NN.Native.Abstraction.Common;
using NN.Native.Abstraction.Data;
using NN.Native.Basic;
using NN.Native.Data;
using NN.Native.Exceptions;
using System.Diagnostics;

namespace NN.Native.Abstraction.Operators
{
    public unsafe interface IMatmulOperator<T>: ITernaryOperator where T : unmanaged
    {
#if NET7_0_OR_GREATER
        static abstract
#endif
        NativeArray<T> Exec(in NativeArray<T> a, in NativeArray<T> b, INativeMemoryManager? memoryManager = null);

        internal static NativeLayout DeduceLayout(ref NativeLayout a, ref NativeLayout b)
        {
            Debug.Assert(a.Ndim <= 2 && b.Ndim <= 2);
            if (a.Ndim <= 0 || b.Ndim <= 0)
            {
                throw new InvalidShapeException();
            }
            if (a.Ndim == 1 && b.Ndim == 1)
            {
                if (a.Shape[0] == 1 && b.Shape[0] == 1)
                {
                    return new NativeLayout(1);
                }
                throw new InvalidShapeException();
            }
            else if (a.Ndim == 2 && b.Ndim == 1)
            {
                if (a.Shape[1] == b.Shape[0])
                {
                    b.AddAxisInplace(1);
                    return new NativeLayout(a.Shape[0], b.Shape[1]);
                }
                else
                {
                    throw new InvalidShapeException("Shape mismatched for matmul.");
                }
            }
            else if (a.Ndim == 1 && b.Ndim == 2)
            {
                if (a.Shape[0] == b.Shape[0])
                {
                    a.AddAxisInplace(0);
                    return new NativeLayout(a.Shape[0], b.Shape[1]);
                }
                else
                {
                    throw new InvalidShapeException("Shape mismatched for matmul.");
                }
            }
            else
            {
                if (a.Shape[1] == b.Shape[0])
                {
                    return new NativeLayout(a.Shape[0], b.Shape[1]);
                }
                else
                {
                    throw new InvalidShapeException("Shape mismatched for matmul.");
                }
            }
        }
    }
}
